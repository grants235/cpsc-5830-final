#!/usr/bin/env python3
"""
Phase 8 experiments (spex8.md): temporal subgraph sampling + ID-GNN.

Usage:
    python scripts/run_phase8.py --exp ts_sage    --delta 60 --seeds 0 1 2
    python scripts/run_phase8.py --exp ts_idgnn   --delta 60 --seeds 0 1 2
    python scripts/run_phase8.py --exp probe      --models ts_sage ts_idgnn
    python scripts/run_phase8.py --exp per_attack --delta 60 [--seed 0]
    python scripts/run_phase8.py --exp pairwise   --delta 60 --seeds 0 1 2
    python scripts/run_phase8.py --exp delta_sweep --seeds 0
    python scripts/run_phase8.py --exp all        --seeds 0 1 2

    Add --no-dev to use full datasets (default: dev subsamples).
"""

import argparse
import csv
import copy
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as _tts
from torch_geometric.data import Batch

from run_phase4 import ALL_FOLDS, E1E_REF, E1E_MEAN, _make_val_split

from src.utils.logging import setup_logging, log_result, already_done, save_model, MODELS_DIR
from src.utils.seeding import seed_everything
from src.utils.metrics import compute_all_metrics, compute_mcc, CLASSES
from src.data.graph_builder import load_graph, combine_graphs
from src.data.temporal_subgraph import batch_build_subgraphs, extract_temporal_subgraph, build_subgraph_data
from src.models.temporal_gnn import TemporalEdgeSAGE, TemporalIDGNN
from src.train.train_loops import _class_weights

log = logging.getLogger(__name__)

FIGURES_DIR  = Path("results/figures/phase8")
DEFAULT_DELTA = 60            # seconds
DELTA_SWEEP  = [10, 60, 300, 1800]
MAX_SUB_EDGES    = 1024
BATCH_SIZE       = 2048       # query edges per batch (large → fewer Python iters)
MAX_TRAIN_EDGES  = 200_000    # cap training edges per epoch (subgraph extraction is expensive)
MAX_EPOCHS       = 20
PATIENCE         = 5
NODE_FEAT_DIM    = 8
N_JOBS           = 1          # subgraph extraction workers (threading overhead > gain for small subgraphs)

PAIRWISE_FOLDS = [
    {"train": ["cic_ids2018"],   "test": "lycos_ids2017"},
    {"train": ["lycos_ids2017"], "test": "cic_ids2018"},
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _delta_us(delta_secs: int) -> int:
    return delta_secs * 1_000_000


def _graph_arrays(graph):
    """Return (src_np, dst_np, time_np) numpy arrays for subgraph extraction."""
    return (
        graph.edge_index[0].numpy(),
        graph.edge_index[1].numpy(),
        graph.edge_time.numpy(),
    )


def _load_fold_struct(fold, dev):
    """Structure-only fold: edge features = constant 1.0 (matching E1.E)."""
    train_graphs = []
    for ds in fold["train"]:
        g = load_graph(ds, tier="B", dev=dev)
        g = copy.copy(g)
        E = g.edge_attr.shape[0]
        g.edge_attr   = torch.ones(E, 1)
        g.edge_attr_q = torch.ones(E, 1)
        train_graphs.append(g)
    combined = combine_graphs(train_graphs)

    g_test = load_graph(fold["test"], tier="B", dev=dev)
    g_test = copy.copy(g_test)
    E = g_test.edge_attr.shape[0]
    g_test.edge_attr   = torch.ones(E, 1)
    g_test.edge_attr_q = torch.ones(E, 1)
    return combined, g_test, fold["train"], fold["test"]


# ── Batch forward pass for temporal models ───────────────────────────────────

def _forward_batch(model, data_list, labels_batch, device):
    """
    Build a PyG Batch from data_list, run the temporal model, return logits.
    labels_batch: [B] tensor (used only for the caller, not here).
    """
    batch = Batch.from_data_list([d.to(device) for d in data_list])
    ptr   = batch.ptr.to(device)          # [B+1] node offsets

    u_globals = torch.tensor(
        [d.query_u for d in data_list], dtype=torch.long, device=device
    ) + ptr[:-1]
    v_globals = torch.tensor(
        [d.query_v for d in data_list], dtype=torch.long, device=device
    ) + ptr[:-1]

    B    = len(data_list)
    q_ea = torch.ones(B, 1, device=device)   # structure-only

    return model(batch.x, batch.edge_index, batch.edge_attr,
                 u_globals, v_globals, q_ea)


@torch.no_grad()
def _embed_batch(model, data_list, device):
    """Like _forward_batch but returns pre-head embeddings [B, 3H]."""
    batch = Batch.from_data_list([d.to(device) for d in data_list])
    ptr   = batch.ptr.to(device)
    u_globals = torch.tensor([d.query_u for d in data_list],
                              dtype=torch.long, device=device) + ptr[:-1]
    v_globals = torch.tensor([d.query_v for d in data_list],
                              dtype=torch.long, device=device) + ptr[:-1]
    q_ea = torch.ones(len(data_list), 1, device=device)
    return model.embed(batch.x, batch.edge_index, batch.edge_attr,
                       u_globals, v_globals, q_ea)


# ── Training loop ─────────────────────────────────────────────────────────────

def _train_temporal(model, graph, device, seed, exp_id, test_dset, delta_us,
                    max_edges=MAX_SUB_EDGES, epochs=MAX_EPOCHS, patience=PATIENCE,
                    batch_size=BATCH_SIZE, max_train_edges=MAX_TRAIN_EDGES):
    """
    Train a temporal GNN (TS-SAGE or TS-IDGNN) on `graph`.
    Returns (model, best_state_dict).

    max_train_edges: stratified subsample of training edges per epoch.
    Subgraph extraction is expensive; iterating over 6M edges per epoch is
    intractable. 200K edges → ~100 batches/epoch, ~1 min/epoch on CPU.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    cw        = _class_weights(graph.edge_label, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    src_np, dst_np, time_np = _graph_arrays(graph)
    labels_np = graph.edge_label.numpy()
    n = len(labels_np)

    # Stratified 80/20 train/val split
    ti, vi = _tts(np.arange(n), test_size=0.2, random_state=seed,
                  stratify=labels_np)
    ti_full = np.array(ti, dtype=np.int64)

    # Cap val set too so validation doesn't dominate wall time
    max_val = max_train_edges // 4
    if len(vi) > max_val:
        rng_val = np.random.RandomState(seed + 999)
        vi = rng_val.choice(vi, max_val, replace=False)
    vi = np.array(vi, dtype=np.int64)

    log.info(f"  train pool={len(ti_full):,}  val={len(vi):,}"
             f"  epoch_cap={min(max_train_edges, len(ti_full)):,}")

    best_mcc, best_state, pat_cnt = -2.0, None, 0
    ep_rng = np.random.RandomState(seed)

    for epoch in range(epochs):
        model.train()
        # Stratified subsample of training edges for this epoch
        if len(ti_full) > max_train_edges:
            ep_labels = labels_np[ti_full]
            ti_arr, _ = _tts(ti_full, test_size=1.0 - max_train_edges / len(ti_full),
                              random_state=ep_rng.randint(0, 2**31),
                              stratify=ep_labels)
            ti_arr = np.array(ti_arr, dtype=np.int64)
        else:
            ti_arr = ti_full.copy()
        ep_rng.shuffle(ti_arr)
        ep_loss, n_batches = 0.0, 0

        for start in range(0, len(ti_arr), batch_size):
            ids   = ti_arr[start:start + batch_size]
            yu_np = src_np[ids]
            yv_np = dst_np[ids]
            yt_np = time_np[ids]
            yl    = torch.as_tensor(labels_np[ids], dtype=torch.long, device=device)

            data_list = batch_build_subgraphs(
                src_np, dst_np, time_np,
                yu_np, yv_np, yt_np,
                delta_us=delta_us, max_edges=max_edges,
                node_feat_dim=NODE_FEAT_DIM, seed=seed,
                n_jobs=N_JOBS,
            )

            logits = _forward_batch(model, data_list, yl, device)
            loss   = criterion(logits, yl)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss  += loss.item()
            n_batches += 1

        # ── Validation ──
        model.eval()
        all_preds = []
        for start in range(0, len(vi), batch_size):
            ids   = vi[start:start + batch_size]  # vi already capped above
            yu_np = src_np[ids]
            yv_np = dst_np[ids]
            yt_np = time_np[ids]
            yl    = torch.as_tensor(labels_np[ids], dtype=torch.long, device=device)
            with torch.no_grad():
                data_list = batch_build_subgraphs(
                    src_np, dst_np, time_np,
                    yu_np, yv_np, yt_np,
                    delta_us=delta_us, max_edges=max_edges,
                    node_feat_dim=NODE_FEAT_DIM, seed=seed,
                    n_jobs=N_JOBS,
                )
                logits = _forward_batch(model, data_list, yl, device)
            all_preds.append(logits.argmax(1).cpu().numpy())

        val_mcc = compute_mcc(labels_np[vi], np.concatenate(all_preds))
        log.info(f"  epoch {epoch+1:02d}  loss={ep_loss/max(1,n_batches):.4f}"
                 f"  val_mcc={val_mcc:.4f}")

        if val_mcc > best_mcc:
            best_mcc   = val_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt    = 0
        else:
            if epoch >= 3:
                pat_cnt += 1
                if pat_cnt >= patience:
                    log.info(f"  Early stop epoch {epoch+1}")
                    break

    log.info(f"  Best val MCC: {best_mcc:.4f}")
    if best_state:
        model.load_state_dict(best_state)
        save_model(exp_id, seed, test_dset, best_state)
    return model, best_state


# ── Evaluation loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_temporal(model, graph, device, delta_us,
                   max_edges=MAX_SUB_EDGES, batch_size=BATCH_SIZE):
    """Evaluate temporal model on a full graph; returns dict with y_true, y_pred."""
    model.eval()
    src_np, dst_np, time_np = _graph_arrays(graph)
    labels_np = graph.edge_label.numpy()
    n = len(labels_np)

    all_preds, all_scores = [], []
    for start in range(0, n, batch_size):
        ids   = np.arange(start, min(start + batch_size, n))
        yu_np = src_np[ids]
        yv_np = dst_np[ids]
        yt_np = time_np[ids]
        yl    = torch.as_tensor(labels_np[ids], dtype=torch.long, device=device)

        data_list = batch_build_subgraphs(
            src_np, dst_np, time_np,
            yu_np, yv_np, yt_np,
            delta_us=delta_us, max_edges=max_edges,
            node_feat_dim=NODE_FEAT_DIM, seed=0,
            n_jobs=N_JOBS,
        )
        logits = _forward_batch(model, data_list, yl, device)
        probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_scores.append(probs)

    return {
        "y_true":  labels_np,
        "y_pred":  np.concatenate(all_preds),
        "y_score": np.concatenate(all_scores),
    }


# ── Probe helper ──────────────────────────────────────────────────────────────

def _probe_on_temporal_encoder(model, train_dsets, dev, seed, device, delta_us,
                                max_edges=MAX_SUB_EDGES, max_per_ds=500):
    """Linear probe: predict source dataset from temporal encoder embeddings."""
    all_embs, all_labels = [], []
    for ds_idx, ds in enumerate(train_dsets):
        g = load_graph(ds, tier="B", dev=dev)
        g = copy.copy(g)
        E = g.edge_attr.shape[0]
        g.edge_attr   = torch.ones(E, 1)
        g.edge_attr_q = torch.ones(E, 1)
        src_np, dst_np, time_np = _graph_arrays(g)

        rng  = np.random.RandomState(seed)
        idx  = rng.choice(E, min(max_per_ds, E), replace=False)
        idx  = np.sort(idx)

        embs = []
        for start in range(0, len(idx), BATCH_SIZE):
            ids   = idx[start:start + BATCH_SIZE]
            data_list = batch_build_subgraphs(
                src_np, dst_np, time_np,
                src_np[ids], dst_np[ids], time_np[ids],
                delta_us=delta_us, max_edges=max_edges,
                node_feat_dim=NODE_FEAT_DIM, seed=seed,
                n_jobs=N_JOBS,
            )
            with torch.no_grad():
                embs.append(_embed_batch(model, data_list, device).cpu().numpy())

        all_embs.append(np.concatenate(embs))
        all_labels.extend([ds_idx] * len(idx))

    X = np.concatenate(all_embs)
    y = np.array(all_labels)
    ti, vi = _tts(np.arange(len(X)), test_size=0.2, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X[ti], y[ti])
    return accuracy_score(y[vi], clf.predict(X[vi]))


# ── Summary helper ────────────────────────────────────────────────────────────

def _print_e8_summary(exp_id, seeds):
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return None
    fold_vals: dict = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["experiment_id"] != exp_id or row["metric"] != "mcc":
                continue
            if int(row["seed"]) in seeds:
                fold_vals.setdefault(row["test_dataset"], []).append(float(row["value"]))
    log.info(f"\n  {exp_id} summary:")
    fold_means = []
    for td, vals in sorted(fold_vals.items()):
        m, s = np.mean(vals), np.std(vals)
        fold_means.append(m)
        log.info(f"    {td:<20} mean={m:.4f}  std={s:.4f}  n={len(vals)}")
    if fold_means:
        overall = np.mean(fold_means)
        log.info(f"  Overall mean MCC: {overall:.4f}  (E1.E baseline: {E1E_MEAN:.4f})")
        if overall > 0.40:
            log.info("  → Strong positive result.")
        elif overall > E1E_MEAN + 0.05:
            log.info(f"  → Beats E1.E by >{overall - E1E_MEAN:.2f}. Temporal sampling helps.")
        elif overall > E1E_MEAN:
            log.info(f"  → Marginal improvement over E1.E.")
        else:
            log.info(f"  → No improvement over E1.E ({E1E_MEAN:.2f}).")
    return fold_means


def _get_best_e8(seeds):
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return "E8.1_ts_sage_d60", -2.0
    candidates: dict = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["metric"] != "mcc" or int(row["seed"]) not in seeds:
                continue
            eid = row["experiment_id"]
            if eid.startswith("E8."):
                candidates.setdefault(eid, []).append(float(row["value"]))
    if not candidates:
        return "E8.1_ts_sage_d60", -2.0
    # Aggregate by fold-mean per experiment
    fold_means = {k: np.mean(v) for k, v in candidates.items()}
    # Average of fold means per experiment
    exp_means: dict = {}
    for k, fvals in candidates.items():
        exp_means[k] = np.mean(fvals)
    best = max(exp_means, key=exp_means.get)
    return best, exp_means[best]


# ── E8.1 — TS-SAGE ────────────────────────────────────────────────────────────

def run_e8_1_ts_sage(seeds, delta_secs: int, dev: bool,
                     folds=None, exp_id_prefix="E8.1_ts_sage"):
    folds  = folds or ALL_FOLDS
    du     = _delta_us(delta_secs)
    exp_id = f"{exp_id_prefix}_d{delta_secs}"
    log.info(f"=== E8.1  TS-SAGE  Δ={delta_secs}s  seeds={seeds} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for seed in seeds:
        for fold in folds:
            train_dsets = fold["train"]
            test_dset   = fold["test"]

            if already_done(exp_id, seed, test_dset):
                log.info(f"  Skipping seed={seed} test={test_dset}")
                continue

            seed_everything(seed)
            t0 = time.time()
            log.info(f"\n  Fold: train={train_dsets}  test={test_dset}  seed={seed}")

            combined, test_graph, _, _ = _load_fold_struct(fold, dev)

            model = TemporalEdgeSAGE(
                node_in=NODE_FEAT_DIM, edge_in=1, hidden=128, num_classes=2
            )
            model, _ = _train_temporal(
                model, combined, device, seed, exp_id, test_dset,
                delta_us=du, epochs=MAX_EPOCHS, patience=PATIENCE,
                batch_size=BATCH_SIZE,
            )

            result  = _eval_temporal(model, test_graph, device, delta_us=du)
            metrics = compute_all_metrics(result["y_true"], result["y_pred"],
                                          y_true_type=test_graph.edge_label_type)
            elapsed = time.time() - t0

            log.info(f"  seed={seed} fold={test_dset}  MCC={metrics['mcc']:.4f}"
                     f"  macro_F1={metrics['macro_f1']:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset, "mcc",
                       metrics["mcc"], elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "macro_f1",
                       metrics["macro_f1"], elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "delta_secs",
                       float(delta_secs), 0.0)

            probe_acc = _probe_on_temporal_encoder(
                model, train_dsets, dev, seed, device, du)
            log.info(f"  Probe accuracy: {probe_acc:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset,
                       "dataset_probe_acc", probe_acc, 0.0)

    _print_e8_summary(exp_id, seeds)


# ── E8.2 — TS-IDGNN ───────────────────────────────────────────────────────────

def run_e8_2_ts_idgnn(seeds, delta_secs: int, dev: bool,
                      folds=None, exp_id_prefix="E8.2_ts_idgnn"):
    folds  = folds or ALL_FOLDS
    du     = _delta_us(delta_secs)
    exp_id = f"{exp_id_prefix}_d{delta_secs}"
    log.info(f"=== E8.2  TS-IDGNN  Δ={delta_secs}s  seeds={seeds} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for seed in seeds:
        for fold in folds:
            train_dsets = fold["train"]
            test_dset   = fold["test"]

            if already_done(exp_id, seed, test_dset):
                log.info(f"  Skipping seed={seed} test={test_dset}")
                continue

            seed_everything(seed)
            t0 = time.time()
            log.info(f"\n  Fold: train={train_dsets}  test={test_dset}  seed={seed}")

            combined, test_graph, _, _ = _load_fold_struct(fold, dev)

            model = TemporalIDGNN(
                node_in=NODE_FEAT_DIM, edge_in=1, hidden=128, num_classes=2
            )
            model, _ = _train_temporal(
                model, combined, device, seed, exp_id, test_dset,
                delta_us=du, epochs=MAX_EPOCHS, patience=PATIENCE,
                batch_size=BATCH_SIZE,
            )

            result  = _eval_temporal(model, test_graph, device, delta_us=du)
            metrics = compute_all_metrics(result["y_true"], result["y_pred"],
                                          y_true_type=test_graph.edge_label_type)
            elapsed = time.time() - t0

            log.info(f"  seed={seed} fold={test_dset}  MCC={metrics['mcc']:.4f}"
                     f"  macro_F1={metrics['macro_f1']:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset, "mcc",
                       metrics["mcc"], elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "macro_f1",
                       metrics["macro_f1"], elapsed)

            probe_acc = _probe_on_temporal_encoder(
                model, train_dsets, dev, seed, device, du)
            log.info(f"  Probe accuracy: {probe_acc:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset,
                       "dataset_probe_acc", probe_acc, 0.0)

    _print_e8_summary(exp_id, seeds)


# ── E8.3 — Probe ─────────────────────────────────────────────────────────────

def run_e8_3_probe(models_to_probe, seed: int, dev: bool, delta_secs: int = DEFAULT_DELTA):
    log.info(f"=== E8.3  Probe  models={models_to_probe}  seed={seed} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    du     = _delta_us(delta_secs)

    model_cls_map = {
        "ts_sage":  (TemporalEdgeSAGE,  f"E8.1_ts_sage_d{delta_secs}"),
        "ts_idgnn": (TemporalIDGNN,     f"E8.2_ts_idgnn_d{delta_secs}"),
    }
    baselines = {
        "raw_flow_features (E6.1)": 0.997,
        "structure_only E1.E":      0.72,
    }

    for key in models_to_probe:
        ModelCls, exp_prefix = model_cls_map.get(key, (TemporalEdgeSAGE, key))
        log.info(f"\n  --- {key} ({exp_prefix}) ---")

        for fold in ALL_FOLDS:
            test_dset   = fold["test"]
            train_dsets = fold["train"]
            enc_path    = MODELS_DIR / f"{exp_prefix}_seed{seed}_test{test_dset}.pt"

            if not enc_path.exists():
                log.warning(f"  Not found: {enc_path}")
                continue

            model = ModelCls(node_in=NODE_FEAT_DIM, edge_in=1, hidden=128)
            model.load_state_dict(torch.load(enc_path, weights_only=True))

            probe_acc = _probe_on_temporal_encoder(
                model, train_dsets, dev, seed, device, du)
            random_b  = 1.0 / len(train_dsets)
            log.info(f"  {key} fold={test_dset}  probe={probe_acc:.4f}"
                     f"  (random={random_b:.2f})")
            log_result(f"E8.3_probe_{key}", seed, train_dsets, test_dset,
                       "dataset_probe_acc", probe_acc, 0.0)

            if probe_acc < 0.50:
                log.info("    → Invariance achieved. Shortcut broken.")
            elif probe_acc < 0.72:
                log.info("    → Partial invariance (below E1.E's 72%).")
            else:
                log.info("    → Leakage persists (>72%).")

    log.info("\n  Baselines:")
    for name, acc in baselines.items():
        log.info(f"    {name}: {acc:.3f}")


# ── E8.4 — Per-attack analysis ────────────────────────────────────────────────

def run_e8_4_per_attack(seed: int, dev: bool, delta_secs: int = DEFAULT_DELTA):
    log.info("=== E8.4  Per-attack analysis ===")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    du     = _delta_us(delta_secs)

    best_exp, best_mean = _get_best_e8([seed])
    log.info(f"  Best E8 method: {best_exp}  mean_mcc={best_mean:.4f}")

    ModelCls = TemporalIDGNN if "idgnn" in best_exp else TemporalEdgeSAGE
    attack_classes = CLASSES[1:]
    results_table: dict = {}

    for fold in ALL_FOLDS:
        train_dsets = fold["train"]
        test_dset   = fold["test"]
        enc_path    = MODELS_DIR / f"{best_exp}_seed{seed}_test{test_dset}.pt"
        if not enc_path.exists():
            log.warning(f"  Missing {enc_path}")
            continue

        model = ModelCls(node_in=NODE_FEAT_DIM, edge_in=1, hidden=128)
        model.load_state_dict(torch.load(enc_path, weights_only=True))

        _, test_graph, _, _ = _load_fold_struct(fold, dev)
        result  = _eval_temporal(model, test_graph, device, delta_us=du)
        metrics = compute_all_metrics(result["y_true"], result["y_pred"],
                                      y_true_type=test_graph.edge_label_type)
        results_table[test_dset] = metrics.get("per_class_f1", {})
        log.info(f"  fold={test_dset}  MCC={metrics['mcc']:.4f}")
        for cls, f1 in metrics.get("per_class_f1", {}).items():
            log_result("E8.4_per_attack", seed, train_dsets, test_dset,
                       f"f1_{cls}", f1, 0.0)

    if not results_table:
        log.warning("  No results.")
        return

    folds_order = [f["test"] for f in ALL_FOLDS]
    mean_f1 = {}
    for cls in attack_classes:
        vals = [results_table.get(d, {}).get(cls, float("nan")) for d in folds_order]
        mean_f1[cls] = float(np.nanmean(vals))

    sorted_cls = sorted(mean_f1, key=mean_f1.get, reverse=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(sorted_cls)), [mean_f1[c] for c in sorted_cls], color="teal")
    ax.set_xticks(range(len(sorted_cls)))
    ax.set_xticklabels(sorted_cls, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean F1 across folds")
    ax.set_ylim(0, 1)
    ax.set_title(f"E8.4  {best_exp}  per-attack mean F1")
    fig.tight_layout()
    out = FIGURES_DIR / "e8_4_per_attack_f1.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    log.info(f"  Saved {out}")

    log.info(f"\n  Per-attack mean F1:")
    for cls in sorted_cls:
        log.info(f"    {cls:<24}  {mean_f1[cls]:.4f}")


# ── E8.5 — Pairwise CIC17 ↔ CIC18 ────────────────────────────────────────────

def run_e8_5_pairwise(seeds, delta_secs: int, dev: bool):
    log.info(f"=== E8.5  Pairwise CIC17↔CIC18  Δ={delta_secs}s  seeds={seeds} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run E8.1 then E8.2 on pairwise folds
    run_e8_1_ts_sage(seeds, delta_secs, dev,
                     folds=PAIRWISE_FOLDS,
                     exp_id_prefix="E8.5_pairwise_sage")
    run_e8_2_ts_idgnn(seeds, delta_secs, dev,
                      folds=PAIRWISE_FOLDS,
                      exp_id_prefix="E8.5_pairwise_idgnn")

    # Print combined summary
    log.info("\n  Pairwise results (CIC17↔CIC18):")
    for exp_pfx in ["E8.5_pairwise_sage", "E8.5_pairwise_idgnn"]:
        exp_id = f"{exp_pfx}_d{delta_secs}"
        _print_e8_summary(exp_id, seeds)


# ── Δ sweep ───────────────────────────────────────────────────────────────────

def run_delta_sweep(seeds, dev: bool):
    log.info(f"=== Δ sweep  deltas={DELTA_SWEEP}  seeds={seeds} ===")
    for d in DELTA_SWEEP:
        log.info(f"\n--- Δ = {d}s ---")
        run_e8_1_ts_sage(seeds, delta_secs=d, dev=dev)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Phase 8 (spex8.md)")
    parser.add_argument("--exp", required=True,
                        choices=["ts_sage", "ts_idgnn", "probe",
                                 "per_attack", "pairwise", "delta_sweep", "all"])
    parser.add_argument("--delta",   type=int,  default=DEFAULT_DELTA,
                        help="Temporal window in seconds (default 60)")
    parser.add_argument("--seeds",   nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seed",    type=int,  default=0)
    parser.add_argument("--models",  nargs="+", default=["ts_sage", "ts_idgnn"])
    parser.add_argument("--dev",    action="store_true", default=True)
    parser.add_argument("--no-dev", dest="dev", action="store_false")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS,
                        help="Subgraph extraction workers (default: 1; try 4-8 on many-core nodes)")
    parser.add_argument("--max-train-edges", type=int, default=MAX_TRAIN_EDGES,
                        help="Training edge cap per epoch (default: 200000)")
    args = parser.parse_args()

    # Allow CLI overrides of globals
    import scripts.run_phase8 as _self
    _self.N_JOBS          = args.n_jobs
    _self.MAX_TRAIN_EDGES = args.max_train_edges

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.exp in ("ts_sage", "all"):
        run_e8_1_ts_sage(args.seeds, args.delta, args.dev)

    if args.exp in ("ts_idgnn", "all"):
        run_e8_2_ts_idgnn(args.seeds, args.delta, args.dev)

    if args.exp in ("probe", "all"):
        run_e8_3_probe(args.models, args.seed, args.dev, args.delta)

    if args.exp in ("per_attack", "all"):
        run_e8_4_per_attack(args.seed, args.dev, args.delta)

    if args.exp in ("pairwise", "all"):
        run_e8_5_pairwise(args.seeds, args.delta, args.dev)

    if args.exp == "delta_sweep":
        run_delta_sweep(args.seeds, args.dev)


if __name__ == "__main__":
    main()
