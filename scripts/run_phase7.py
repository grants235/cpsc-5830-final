#!/usr/bin/env python3
"""
Phase 7 experiments (spex7.md): local reference frames for cross-dataset NIDS.

Usage:
    python scripts/run_phase7.py --exp lqe          --feature_set tier_a --seeds 0 1 2
    python scripts/run_phase7.py --exp lqe          --feature_set tier_b --seeds 0 1 2
    python scripts/run_phase7.py --exp lze           --seeds 0 1 2
    python scripts/run_phase7.py --exp probe         --models lqe lze
    python scripts/run_phase7.py --exp lqe_hybrid    --seeds 0 1 2
    python scripts/run_phase7.py --exp per_attack    [--seed 0]
    python scripts/run_phase7.py --exp lqe_tgn       --seeds 0 1 2
    python scripts/run_phase7.py --exp all           [--seeds 0 1 2]

    Add --no-dev to use full datasets (default: dev subsamples).
"""

import argparse
import copy
import csv
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as _tts

from run_phase4 import (
    ALL_FOLDS, E1E_REF, E1E_MEAN,
    _make_val_split,
)
from run_phase6 import (
    ANOMALY_EDGE_IN,
    _index_subgraph,
    _extract_embeddings,
    _get_anomaly_scores,
    _determine_best_e6,
)

from src.utils.logging import setup_logging, log_result, already_done, save_model, MODELS_DIR
from src.utils.seeding import seed_everything
from src.utils.metrics import compute_all_metrics, compute_mcc, CLASSES
from src.data.graph_builder import load_graph, combine_graphs, quantile_encode, PROCESSED_DIR
from src.models.egraphsage import EdgeAwareSAGE
from src.models.tgn_ids import TGN_IDS
from src.train.eval import eval_egraphsage, eval_tgn
from src.train.train_loops import _class_weights, train_tgn

log = logging.getLogger(__name__)

FIGURES_DIR = Path("results/figures/phase7")
LQE_K       = 64


# ── LQE / LZE computation ────────────────────────────────────────────────────

def compute_lqe(src_nodes: np.ndarray, edge_attr: np.ndarray, k: int = LQE_K) -> np.ndarray:
    """
    Local quantile encoding: percentile rank within per-source temporal window.
    Edges must be in temporal order (ascending timestamp).
    Returns float32 array in [0, 1], same shape as edge_attr.
    """
    E, F = edge_attr.shape
    out = np.full((E, F), 0.5, dtype=np.float32)

    node_edges: dict = defaultdict(list)
    for i in range(E):
        node_edges[int(src_nodes[i])].append(i)

    for indices in node_edges.values():
        n = len(indices)
        feat = edge_attr[indices]  # [n, F] temporal order
        for j in range(1, n):
            w = feat[max(0, j - k):j]       # [W, F]  W <= k
            W = w.shape[0]
            v = feat[j]                      # [F]
            n_less = (w < v).sum(axis=0).astype(np.float32)
            n_eq   = (w == v).sum(axis=0).astype(np.float32)
            out[indices[j]] = (n_less + 0.5 * n_eq) / W

    return out


def compute_lze(src_nodes: np.ndarray, edge_attr: np.ndarray,
                k: int = LQE_K, eps: float = 1e-8) -> np.ndarray:
    """
    Local z-score encoding: (value - local_mean) / (local_std + eps).
    Returns float32 array (unbounded).
    """
    E, F = edge_attr.shape
    out = np.zeros((E, F), dtype=np.float32)

    node_edges: dict = defaultdict(list)
    for i in range(E):
        node_edges[int(src_nodes[i])].append(i)

    for indices in node_edges.values():
        n = len(indices)
        feat = edge_attr[indices]   # [n, F]
        for j in range(1, n):
            w = feat[max(0, j - k):j]
            mu    = w.mean(axis=0)
            sigma = w.std(axis=0)
            out[indices[j]] = (feat[j] - mu) / (sigma + eps)

    return out


# ── Graph building with LQE/LZE caching ──────────────────────────────────────

def _build_lqe_graph(dataset_name: str, tier: str, dev: bool,
                     kind: str = "lqe", k: int = LQE_K):
    """
    Load raw graph, compute LQE or LZE features, cache result.
    Returns a Data object where edge_attr = encoded features and
    edge_attr_q = those features (LQE already [0,1]; LZE quantile-encoded).
    """
    suffix     = "_dev" if dev else "_full"
    cache_path = PROCESSED_DIR / f"{dataset_name}_tier{tier}_{kind}{suffix}.pt"

    if cache_path.exists():
        log.info(f"  Loading cached {kind}: {cache_path}")
        return torch.load(cache_path, weights_only=False)

    log.info(f"  Computing {kind} for {dataset_name} tier={tier} dev={dev} ...")
    t0 = time.time()
    g  = load_graph(dataset_name, tier=tier, dev=dev)

    src_np  = g.edge_index[0].numpy()
    feat_np = g.edge_attr.numpy()

    if kind == "lqe":
        enc_np  = compute_lqe(src_np, feat_np, k=k)
        enc_t   = torch.as_tensor(enc_np, dtype=torch.float32)
        enc_q   = enc_t.clone()            # already in [0, 1]
    else:
        enc_np  = compute_lze(src_np, feat_np, k=k)
        enc_t   = torch.as_tensor(enc_np, dtype=torch.float32)
        enc_q   = quantile_encode(enc_t)   # map z-scores → [0, 1]

    g2 = copy.copy(g)
    g2.edge_attr   = enc_t
    g2.edge_attr_q = enc_q

    torch.save(g2, cache_path)
    log.info(f"  Saved {cache_path}  ({time.time()-t0:.1f}s)")
    return g2


def _load_fold_lqe(fold: dict, dev: bool, tier: str = "A", kind: str = "lqe"):
    """Load LQE/LZE combined training + test graphs for a fold."""
    train_dsets  = fold["train"]
    test_dset    = fold["test"]
    train_graphs = [_build_lqe_graph(ds, tier, dev, kind) for ds in train_dsets]
    combined     = combine_graphs(train_graphs)
    test_graph   = _build_lqe_graph(test_dset, tier, dev, kind)

    # Align feature dims for Tier-B (may differ across datasets)
    max_feat = combined.edge_attr.shape[1]
    d        = test_graph.edge_attr.shape[1]
    if d < max_feat:
        pad = torch.zeros(test_graph.edge_attr.shape[0], max_feat - d)
        test_graph.edge_attr   = torch.cat([test_graph.edge_attr, pad], dim=1)
        test_graph.edge_attr_q = torch.cat([test_graph.edge_attr_q, pad], dim=1)
    elif d > max_feat:
        test_graph.edge_attr   = test_graph.edge_attr[:, :max_feat]
        test_graph.edge_attr_q = test_graph.edge_attr_q[:, :max_feat]

    return combined, test_graph, train_dsets, test_dset


# ── Training helper (seed-correct saving) ────────────────────────────────────

def _train_lqe_model(graph, device, seed, exp_id, test_dset,
                     epochs=30, patience=5, batch_size=2048):
    """Train EdgeAwareSAGE on LQE/LZE graph; save checkpoint with correct seed."""
    model = EdgeAwareSAGE(
        node_in=graph.x.shape[1],
        edge_in=graph.edge_attr.shape[1],
        hidden=128, num_classes=2, dropout=0.2,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    cw        = _class_weights(graph.edge_label, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    n = graph.edge_label.shape[0]
    ti, vi = _tts(np.arange(n), test_size=0.2, random_state=seed,
                  stratify=graph.edge_label.numpy())
    ti_arr = np.array(ti, dtype=np.int64)

    x  = graph.x.to(device)
    ei = graph.edge_index.to(device)
    ea = graph.edge_attr_q.to(device)

    best_mcc, best_state, pat_cnt = -2.0, None, 0

    for epoch in range(epochs):
        model.train()
        np.random.shuffle(ti_arr)
        ep_loss = 0.0
        for s in range(0, len(ti_arr), batch_size):
            ids    = ti_arr[s:s + batch_size]
            logits = model(x, ei[:, ids], ea[ids])
            loss   = criterion(logits, graph.edge_label[ids].to(device))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        model.eval()
        with torch.no_grad():
            preds = []
            for s in range(0, len(vi), batch_size):
                ids = vi[s:s + batch_size]
                preds.append(model(x, ei[:, ids], ea[ids]).argmax(1).cpu().numpy())
        val_mcc = compute_mcc(graph.edge_label[vi].numpy(), np.concatenate(preds))
        log.info(f"  epoch {epoch+1:02d}  loss={ep_loss/max(1, len(ti_arr)//batch_size):.4f}"
                 f"  val_mcc={val_mcc:.4f}")

        if val_mcc > best_mcc:
            best_mcc   = val_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt    = 0
        else:
            if epoch >= 5:
                pat_cnt += 1
                if pat_cnt >= patience:
                    log.info(f"  Early stop epoch {epoch+1}")
                    break

    log.info(f"  Best val MCC: {best_mcc:.4f}")
    if best_state:
        model.load_state_dict(best_state)
        save_model(exp_id, seed, test_dset, best_state)
    return model, best_state


# ── Probe helper ──────────────────────────────────────────────────────────────

def _probe_on_lqe_encoder(model, train_dsets, dev, seed, device,
                          tier="A", kind="lqe", max_per_ds=10_000):
    """Linear probe: predict source dataset from encoder embeddings."""
    all_embs, all_labels = [], []
    for ds_idx, ds in enumerate(train_dsets):
        g    = _build_lqe_graph(ds, tier, dev, kind)
        embs = _extract_embeddings(model, g, device)
        rng  = np.random.RandomState(seed)
        idx  = rng.choice(len(embs), min(max_per_ds, len(embs)), replace=False)
        all_embs.append(embs[idx])
        all_labels.extend([ds_idx] * len(idx))
    X = np.concatenate(all_embs)
    y = np.array(all_labels)
    ti, vi = _tts(np.arange(len(X)), test_size=0.2, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
    clf.fit(X[ti], y[ti])
    return accuracy_score(y[vi], clf.predict(X[vi]))


# ── Summary helpers ───────────────────────────────────────────────────────────

def _print_e7_summary(exp_id, seeds):
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return
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
        e1e     = E1E_MEAN
        log.info(f"  Overall mean MCC: {overall:.4f}  (E1.E baseline: {e1e:.4f})")
        if overall > 0.40:
            log.info("  → Strong positive result. Headline paper result.")
        elif overall > e1e + 0.05:
            log.info(f"  → Beats E1.E by >{overall - e1e:.2f}. Local ref frames help.")
        elif overall > e1e:
            log.info(f"  → Marginal improvement over E1.E ({e1e:.2f}).")
        else:
            log.info(f"  → No improvement over E1.E ({e1e:.2f}).")
    return fold_means


def _determine_best_e7(seeds):
    """Return (exp_id, mean_mcc) of best E7 method."""
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return "E7.1_lqe_tierA", -2.0
    candidates = ["E7.1_lqe_tierA", "E7.2_lze", "E7.4_lqe_hybrid", "E7.6_lqe_tgn"]
    fold_vals: dict = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["metric"] != "mcc" or int(row["seed"]) not in seeds:
                continue
            eid = row["experiment_id"]
            if eid in candidates:
                fold_vals.setdefault(eid, []).append(float(row["value"]))
    if not fold_vals:
        return "E7.1_lqe_tierA", -2.0
    best = max(fold_vals, key=lambda k: np.mean(fold_vals[k]))
    mean = float(np.mean(fold_vals[best]))
    log.info(f"  Best E7 method: {best}  mean_mcc={mean:.4f}")
    return best, mean


# ── E7.1 — Local Quantile Encoding ───────────────────────────────────────────

def run_e7_1_lqe(seeds, feature_set: str, dev: bool):
    tier   = "A" if feature_set == "tier_a" else "B"
    exp_id = f"E7.1_lqe_tier{tier}"
    log.info(f"=== E7.1  LQE tier={tier}  seeds={seeds} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for seed in seeds:
        for fold in ALL_FOLDS:
            train_dsets = fold["train"]
            test_dset   = fold["test"]

            if already_done(exp_id, seed, test_dset):
                log.info(f"  Skipping seed={seed} test={test_dset}")
                continue

            seed_everything(seed)
            t0 = time.time()
            log.info(f"\n  Fold: train={train_dsets}  test={test_dset}  seed={seed}")

            combined, test_graph, _, _ = _load_fold_lqe(fold, dev, tier=tier, kind="lqe")

            model, _ = _train_lqe_model(
                combined, device, seed, exp_id, test_dset,
                epochs=30, patience=5, batch_size=2048,
            )

            result  = eval_egraphsage(model, test_graph, device=device, use_quantile=True)
            metrics = compute_all_metrics(result["y_true"], result["y_pred"],
                                          y_true_type=test_graph.edge_label_type)
            elapsed = time.time() - t0

            log.info(f"  seed={seed} fold={test_dset}  MCC={metrics['mcc']:.4f}"
                     f"  macro_F1={metrics['macro_f1']:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset, "mcc",      metrics["mcc"],      elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "macro_f1", metrics["macro_f1"], elapsed)

            # Probe accuracy — how much dataset identity remains in embeddings
            probe_acc = _probe_on_lqe_encoder(model, train_dsets, dev, seed, device,
                                               tier=tier, kind="lqe")
            log.info(f"  Probe accuracy: {probe_acc:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset, "dataset_probe_acc", probe_acc, 0.0)

    _print_e7_summary(exp_id, seeds)


# ── E7.2 — Local Z-Score Encoding ────────────────────────────────────────────

def run_e7_2_lze(seeds, dev: bool):
    exp_id = "E7.2_lze"
    log.info(f"=== E7.2  LZE  seeds={seeds} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for seed in seeds:
        for fold in ALL_FOLDS:
            train_dsets = fold["train"]
            test_dset   = fold["test"]

            if already_done(exp_id, seed, test_dset):
                log.info(f"  Skipping seed={seed} test={test_dset}")
                continue

            seed_everything(seed)
            t0 = time.time()
            log.info(f"\n  Fold: train={train_dsets}  test={test_dset}  seed={seed}")

            combined, test_graph, _, _ = _load_fold_lqe(fold, dev, tier="A", kind="lze")

            model, _ = _train_lqe_model(
                combined, device, seed, exp_id, test_dset,
                epochs=30, patience=5, batch_size=2048,
            )

            result  = eval_egraphsage(model, test_graph, device=device, use_quantile=True)
            metrics = compute_all_metrics(result["y_true"], result["y_pred"],
                                          y_true_type=test_graph.edge_label_type)
            elapsed = time.time() - t0

            log.info(f"  seed={seed} fold={test_dset}  MCC={metrics['mcc']:.4f}"
                     f"  macro_F1={metrics['macro_f1']:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset, "mcc",      metrics["mcc"],      elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "macro_f1", metrics["macro_f1"], elapsed)

            probe_acc = _probe_on_lqe_encoder(model, train_dsets, dev, seed, device,
                                               tier="A", kind="lze")
            log.info(f"  Probe accuracy: {probe_acc:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset, "dataset_probe_acc", probe_acc, 0.0)

    _print_e7_summary(exp_id, seeds)


# ── E7.3 — Linear probe on LQE/LZE encoders ──────────────────────────────────

def run_e7_3_probe(models_to_probe, seed: int, dev: bool):
    log.info(f"=== E7.3  Linear probe  models={models_to_probe}  seed={seed} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_map = {
        "lqe": ("E7.1_lqe_tierA", "lqe"),
        "lze": ("E7.2_lze",       "lze"),
    }
    baselines = {
        "raw_flow (E6.1)": 0.997,
        "structure_only (E1.E)": 0.72,
    }

    for model_key in models_to_probe:
        exp_prefix, kind = exp_map.get(model_key, (model_key, "lqe"))
        log.info(f"\n  --- Model: {model_key} ({exp_prefix}) ---")

        for fold in ALL_FOLDS:
            test_dset   = fold["test"]
            train_dsets = fold["train"]
            enc_path    = MODELS_DIR / f"{exp_prefix}_seed{seed}_test{test_dset}.pt"

            if not enc_path.exists():
                log.warning(f"  Encoder not found: {enc_path}  (run E7.1/E7.2 first)")
                continue

            encoder = EdgeAwareSAGE(node_in=8, edge_in=4, hidden=128)
            encoder.load_state_dict(torch.load(enc_path, weights_only=True))

            probe_acc = _probe_on_lqe_encoder(encoder, train_dsets, dev, seed, device,
                                               tier="A", kind=kind)
            random_b  = 1.0 / len(train_dsets)
            log.info(f"  {model_key} fold={test_dset}  probe_acc={probe_acc:.4f}"
                     f"  (random={random_b:.2f})")

            log_result(f"E7.3_probe_{model_key}", seed, train_dsets, test_dset,
                       "dataset_probe_acc", probe_acc, 0.0)

            if probe_acc < 0.50:
                log.info("    → Invariance achieved (<50%). Mechanism confirmed.")
            elif probe_acc < 0.70:
                log.info("    → Partial invariance (50–70%). Publishable.")
            else:
                log.info("    → Leakage persists (>70%). Encoding insufficient.")

    log.info("\n  Baselines for reference:")
    for name, acc in baselines.items():
        log.info(f"    {name}: {acc:.3f}")


# ── E7.4 — LQE-Hybrid (LQE features + E6.2 anomaly score) ───────────────────

def _build_hybrid_lqe_graph(lqe_graph, anomaly_scores_np):
    """
    Combine LQE edge features with a normalized anomaly score.
    Returns Data with edge_attr = [LQE_features, norm_score],  shape [E, F+1].
    """
    scores_t = torch.as_tensor(anomaly_scores_np, dtype=torch.float32)
    s_min, s_max = scores_t.min(), scores_t.max()
    scores_norm  = (scores_t - s_min) / (s_max - s_min + 1e-8)
    new_ea  = torch.cat([lqe_graph.edge_attr, scores_norm.unsqueeze(1)], dim=1)
    new_eaq = torch.cat([lqe_graph.edge_attr_q, scores_norm.unsqueeze(1)], dim=1)
    g2 = copy.copy(lqe_graph)
    g2.edge_attr   = new_ea
    g2.edge_attr_q = new_eaq
    return g2


def run_e7_4_hybrid(seeds, dev: bool):
    exp_id = "E7.4_lqe_hybrid"
    log.info(f"=== E7.4  LQE-Hybrid  seeds={seeds} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine which E6 anomaly encoder to use
    best_e6 = _determine_best_e6(seeds)
    log.info(f"  Anomaly base: {best_e6}")

    for seed in seeds:
        for fold in ALL_FOLDS:
            train_dsets = fold["train"]
            test_dset   = fold["test"]

            if already_done(exp_id, seed, test_dset):
                log.info(f"  Skipping seed={seed} test={test_dset}")
                continue

            enc_path = MODELS_DIR / f"{best_e6}_encoder_seed{seed}_test{test_dset}.pt"
            if_path  = MODELS_DIR / f"{best_e6}_if_seed{seed}_test{test_dset}.pt"
            if not enc_path.exists() or not if_path.exists():
                log.warning(f"  Missing E6 checkpoints for seed={seed} test={test_dset}"
                            f" — run Phase 6 first")
                continue

            seed_everything(seed)
            t0 = time.time()
            log.info(f"\n  Fold: train={train_dsets}  test={test_dset}  seed={seed}")

            # Load frozen E6 anomaly encoder + IF
            anomaly_enc = EdgeAwareSAGE(node_in=8, edge_in=ANOMALY_EDGE_IN, hidden=128)
            anomaly_enc.load_state_dict(torch.load(enc_path, weights_only=True))
            iforest = torch.load(if_path, weights_only=False)

            # Load Tier-A raw graphs for anomaly scoring (E6 was trained on Tier-A)
            from src.data.graph_builder import load_graph as _lg
            raw_train_graphs = [_lg(ds, tier="A", dev=dev) for ds in train_dsets]
            from run_phase4 import _make_val_split
            from src.data.graph_builder import combine_graphs as _cg
            raw_combined = _cg(raw_train_graphs)
            raw_test     = _lg(test_dset, tier="A", dev=dev)

            # Load LQE Tier-A graphs (same edge order, just different features)
            lqe_combined  = _load_fold_lqe(fold, dev, tier="A", kind="lqe")[0]
            lqe_test      = _build_lqe_graph(test_dset, "A", dev, kind="lqe")

            # Compute anomaly scores on raw Tier-A features (E6 input space)
            train_scores = _get_anomaly_scores(iforest, anomaly_enc, raw_combined, device)
            test_scores  = _get_anomaly_scores(iforest, anomaly_enc, raw_test,     device)

            # Build hybrid: [LQE_4dim, anomaly_score_1dim] = 5-dim
            train_hyb = _build_hybrid_lqe_graph(lqe_combined, train_scores)
            test_hyb  = _build_hybrid_lqe_graph(lqe_test,     test_scores)

            model, _ = _train_lqe_model(
                train_hyb, device, seed, exp_id, test_dset,
                epochs=30, patience=5, batch_size=2048,
            )

            result  = eval_egraphsage(model, test_hyb, device=device, use_quantile=True)
            metrics = compute_all_metrics(result["y_true"], result["y_pred"],
                                          y_true_type=test_hyb.edge_label_type)
            elapsed = time.time() - t0

            log.info(f"  seed={seed} fold={test_dset}  MCC={metrics['mcc']:.4f}"
                     f"  macro_F1={metrics['macro_f1']:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset, "mcc",      metrics["mcc"],      elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "macro_f1", metrics["macro_f1"], elapsed)

            # Probe: does the hybrid encoder still leak dataset identity?
            probe_acc = _probe_on_lqe_encoder(model, train_dsets, dev, seed, device,
                                               tier="A", kind="lqe")
            log.info(f"  Hybrid probe accuracy: {probe_acc:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset, "dataset_probe_acc", probe_acc, 0.0)

    _print_e7_summary(exp_id, seeds)


# ── E7.5 — Per-attack analysis on best E7 method ─────────────────────────────

def run_e7_5_per_attack(seed: int, dev: bool):
    log.info("=== E7.5  Per-attack analysis on best E7 method ===")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    best_exp, best_mcc_val = _determine_best_e7([seed])
    log.info(f"  Using method: {best_exp}  (mean MCC={best_mcc_val:.4f})")

    # Determine kind and tier from best_exp
    if "hybrid" in best_exp:
        kind, tier = "hybrid", "A"
    elif "lze" in best_exp:
        kind, tier = "lze", "A"
    elif "lqe_tgn" in best_exp:
        kind, tier = "lqe", "A"
    else:
        kind, tier = "lqe", "A"

    attack_classes = CLASSES[1:]
    results_table: dict = {}

    for fold in ALL_FOLDS:
        train_dsets = fold["train"]
        test_dset   = fold["test"]

        enc_path = MODELS_DIR / f"{best_exp}_seed{seed}_test{test_dset}.pt"
        if not enc_path.exists():
            log.warning(f"  Missing checkpoint for fold={test_dset} — skipping")
            continue

        # Determine edge_in from experiment
        if "hybrid" in best_exp:
            edge_in = 5   # 4 LQE + 1 anomaly score
        else:
            edge_in = 4   # Tier-A LQE/LZE

        model = EdgeAwareSAGE(node_in=8, edge_in=edge_in, hidden=128)
        model.load_state_dict(torch.load(enc_path, weights_only=True))

        if "hybrid" in best_exp:
            # Rebuild hybrid test graph
            best_e6 = _determine_best_e6([seed])
            e6_enc_path = MODELS_DIR / f"{best_e6}_encoder_seed{seed}_test{test_dset}.pt"
            e6_if_path  = MODELS_DIR / f"{best_e6}_if_seed{seed}_test{test_dset}.pt"
            if not e6_enc_path.exists():
                log.warning(f"  Missing E6 checkpoint for fold={test_dset} — skipping")
                continue
            from src.data.graph_builder import load_graph as _lg
            anomaly_enc = EdgeAwareSAGE(node_in=8, edge_in=ANOMALY_EDGE_IN, hidden=128)
            anomaly_enc.load_state_dict(torch.load(e6_enc_path, weights_only=True))
            iforest     = torch.load(e6_if_path, weights_only=False)
            raw_test    = _lg(test_dset, tier="A", dev=dev)
            lqe_test    = _build_lqe_graph(test_dset, "A", dev, kind="lqe")
            test_scores = _get_anomaly_scores(iforest, anomaly_enc, raw_test, device)
            test_graph  = _build_hybrid_lqe_graph(lqe_test, test_scores)
        else:
            _, test_graph, _, _ = _load_fold_lqe(fold, dev, tier=tier, kind=kind)

        result  = eval_egraphsage(model, test_graph, device=device, use_quantile=True)
        metrics = compute_all_metrics(result["y_true"], result["y_pred"],
                                      y_true_type=test_graph.edge_label_type)
        results_table[test_dset] = metrics.get("per_class_f1", {})
        log.info(f"  fold={test_dset}  MCC={metrics['mcc']:.4f}")

        for cls, f1 in metrics.get("per_class_f1", {}).items():
            log_result("E7.5_per_attack", seed, train_dsets, test_dset, f"f1_{cls}", f1, 0.0)

    if not results_table:
        log.warning("  No results — run E7.1 or E7.4 first")
        return

    folds_order = [f["test"] for f in ALL_FOLDS]
    mean_f1 = {}
    for cls in attack_classes:
        vals = [results_table.get(d, {}).get(cls, float("nan")) for d in folds_order]
        mean_f1[cls] = float(np.nanmean(vals))

    sorted_cls = sorted(mean_f1, key=mean_f1.get, reverse=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(sorted_cls)), [mean_f1[c] for c in sorted_cls], color="steelblue")
    ax.set_xticks(range(len(sorted_cls)))
    ax.set_xticklabels(sorted_cls, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean F1 across folds")
    ax.set_ylim(0, 1)
    ax.set_title(f"E7.5  {best_exp}  per-attack mean F1")
    fig.tight_layout()
    out = FIGURES_DIR / "e7_5_per_attack_f1.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    log.info(f"  Saved {out}")

    log.info(f"\n  Per-attack mean F1 ({best_exp}):")
    for cls in sorted_cls:
        log.info(f"    {cls:<24}  {mean_f1[cls]:.4f}")

    # Flag volume-based vs low-volume split
    vol_based  = {"DoS_DDoS", "Reconnaissance", "Botnet_C2"}
    low_vol    = {"Injection_Exploit", "BruteForce"}
    vol_f1     = np.nanmean([mean_f1[c] for c in vol_based  if c in mean_f1])
    lowvol_f1  = np.nanmean([mean_f1[c] for c in low_vol    if c in mean_f1])
    log.info(f"\n  Volume-based attacks (DoS/Recon/Botnet)  mean F1={vol_f1:.4f}")
    log.info(f"  Low-volume attacks (Injection/BruteForce) mean F1={lowvol_f1:.4f}")
    if vol_f1 > lowvol_f1 + 0.05:
        log.info("  → LQE concentrates gains on volume-based attacks. Mechanism confirmed.")
    elif lowvol_f1 > vol_f1 + 0.05:
        log.info("  → Unexpectedly better on low-volume attacks.")
    else:
        log.info("  → No clear volume-based concentration.")


# ── E7.6 — LQE on TGN backbone ───────────────────────────────────────────────

def run_e7_6_lqe_tgn(seeds, dev: bool):
    exp_id = "E7.6_lqe_tgn"
    log.info(f"=== E7.6  LQE-TGN  seeds={seeds} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Conditional: only run if E7.1 beats E1.E
    lqe_means = _get_fold_mean("E7.1_lqe_tierA", seeds)
    if lqe_means is not None and lqe_means <= E1E_MEAN:
        log.info(f"  E7.1 mean MCC ({lqe_means:.4f}) ≤ E1.E ({E1E_MEAN:.4f}). "
                 "Skipping E7.6 (condition: LQE must beat E1.E).")
        return
    elif lqe_means is None:
        log.info("  E7.1 results not found — proceeding anyway (conditional skip disabled).")

    for seed in seeds:
        for fold in ALL_FOLDS:
            train_dsets = fold["train"]
            test_dset   = fold["test"]

            if already_done(exp_id, seed, test_dset):
                log.info(f"  Skipping seed={seed} test={test_dset}")
                continue

            seed_everything(seed)
            t0 = time.time()
            log.info(f"\n  Fold: train={train_dsets}  test={test_dset}  seed={seed}")

            combined, test_graph, _, _ = _load_fold_lqe(fold, dev, tier="A", kind="lqe")

            # Temporal 80/20 split for TGN validation
            n          = combined.edge_label.shape[0]
            split_idx  = int(n * 0.8)
            train_data = _index_subgraph(combined, list(range(split_idx)))
            val_data   = _index_subgraph(combined, list(range(split_idx, n)))

            edge_in = combined.edge_attr.shape[1]   # 4 for Tier-A LQE
            model   = TGN_IDS(
                num_nodes   = combined.num_nodes,
                raw_msg_dim = edge_in,
                memory_dim  = 100,
                time_dim    = 100,
                embed_dim   = 100,
            )

            best_state = train_tgn(
                model, train_data, val_data, device=device,
                epochs=30, patience=5, min_epochs=5,
                batch_size=200, use_quantile=True,
            )
            model.load_state_dict(best_state)

            from torch_geometric.nn.models.tgn import LastNeighborLoader
            test_nl   = LastNeighborLoader(model.num_nodes, size=10, device=device)
            test_assoc = torch.empty(model.num_nodes, dtype=torch.long, device=device)
            result    = eval_tgn(model, test_graph, test_nl, test_assoc, device,
                                 batch_size=200, use_quantile=True)
            metrics   = compute_all_metrics(result["y_true"], result["y_pred"],
                                            y_true_type=test_graph.edge_label_type)
            elapsed   = time.time() - t0

            log.info(f"  seed={seed} fold={test_dset}  MCC={metrics['mcc']:.4f}"
                     f"  macro_F1={metrics['macro_f1']:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset, "mcc",      metrics["mcc"],      elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "macro_f1", metrics["macro_f1"], elapsed)

            if best_state:
                save_model(exp_id, seed, test_dset, best_state)

    _print_e7_summary(exp_id, seeds)


def _get_fold_mean(exp_id: str, seeds) -> float | None:
    """Return mean MCC across all folds and seeds for an experiment, or None if missing."""
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return None
    vals = []
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["experiment_id"] == exp_id and row["metric"] == "mcc":
                if int(row["seed"]) in seeds:
                    vals.append(float(row["value"]))
    if not vals:
        return None
    fold_means = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["experiment_id"] == exp_id and row["metric"] == "mcc":
                if int(row["seed"]) in seeds:
                    fold_means.setdefault(row["test_dataset"], []).append(float(row["value"]))
    if not fold_means:
        return None
    return float(np.mean([np.mean(v) for v in fold_means.values()]))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Phase 7 experiments (spex7.md)")
    parser.add_argument("--exp", required=True,
                        choices=["lqe", "lze", "probe", "lqe_hybrid",
                                 "per_attack", "lqe_tgn", "all"])
    parser.add_argument("--feature_set", default="tier_a",
                        choices=["tier_a", "tier_b"],
                        help="Feature tier for LQE (default: tier_a)")
    parser.add_argument("--seeds",  nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seed",   type=int,  default=0,
                        help="Single seed for probe / per_attack")
    parser.add_argument("--models", nargs="+", default=["lqe", "lze"],
                        help="Models to probe (--exp probe)")
    parser.add_argument("--dev",    action="store_true", default=True)
    parser.add_argument("--no-dev", dest="dev", action="store_false")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.exp in ("lqe", "all"):
        run_e7_1_lqe(args.seeds, args.feature_set, args.dev)
        if args.exp == "all" and args.feature_set == "tier_a":
            # Also run Tier-B as part of "all"
            run_e7_1_lqe(args.seeds, "tier_b", args.dev)

    if args.exp in ("lze", "all"):
        run_e7_2_lze(args.seeds, args.dev)

    if args.exp in ("probe", "all"):
        run_e7_3_probe(args.models, args.seed, args.dev)

    if args.exp in ("lqe_hybrid", "all"):
        run_e7_4_hybrid(args.seeds, args.dev)

    if args.exp in ("per_attack", "all"):
        run_e7_5_per_attack(args.seed, args.dev)

    if args.exp in ("lqe_tgn", "all"):
        run_e7_6_lqe_tgn(args.seeds, args.dev)


if __name__ == "__main__":
    main()
