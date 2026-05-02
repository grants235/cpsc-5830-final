#!/usr/bin/env python3
"""
main.py  —  Cross-dataset GNN intrusion detection (LODO evaluation).

Train and evaluate any of the four main model variants on the 4-dataset
Leave-One-Dataset-Out (LODO) benchmark.  Results are appended to
results/results.csv.  Already-completed runs are skipped automatically.

Models
------
  egraphsage   Edge-aware GraphSAGE (Lo et al. 2022)
  gib          GIB-EGraphSAGE with variational information bottleneck
  ts-sage      Temporal-Subgraph SAGE (structure-only)
  ts-gib       Temporal-Subgraph GIB (structure-only + optional bottleneck)

Examples
--------
  python main.py --model egraphsage
  python main.py --model egraphsage --no-features
  python main.py --model gib --beta 0.01
  python main.py --model gib --beta 0.01 0.1 1.0
  python main.py --model ts-sage --no-dev
  python main.py --model ts-gib --beta 0.01 --seeds 0 1 2 --no-dev
  python main.py --model ts-gib --beta 0.01 --folds lycos_ids2017 unsw_nb15
"""

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.logging import setup_logging, log_result, save_model, already_done
from src.utils.seeding import seed_everything
from src.utils.metrics import compute_all_metrics, compute_mcc
from src.data.graph_builder import load_graph, combine_graphs
from src.models.egraphsage import EdgeAwareSAGE
from src.models.gib_egraphsage import GIB_EGraphSAGE
from src.models.temporal_gnn import TS_GIB
from src.data.temporal_subgraph import batch_build_subgraphs
from src.train.train_loops import train_egraphsage, _class_weights
from src.train.eval import eval_egraphsage

log = logging.getLogger(__name__)

NODE_FEAT_DIM = 8  # all-ones node feature dimension used in all models

ALL_FOLDS = [
    {"train": ["cic_ids2018",   "unsw_nb15",   "ton_iot"],       "test": "lycos_ids2017"},
    {"train": ["lycos_ids2017", "unsw_nb15",   "ton_iot"],       "test": "cic_ids2018"},
    {"train": ["lycos_ids2017", "cic_ids2018", "ton_iot"],       "test": "unsw_nb15"},
    {"train": ["lycos_ids2017", "cic_ids2018", "unsw_nb15"],     "test": "ton_iot"},
]


# ── data loading ──────────────────────────────────────────────────────────────

def _load_static(fold: dict, dev: bool, no_features: bool):
    """
    Load and align train + test graphs for E-GraphSAGE / GIB.

    Feature dimensions are padded/trimmed so the test graph matches the
    combined training graph.  When no_features=True all feature values are
    replaced with 1.0 (structure-only ablation).
    """
    train_graphs = [load_graph(ds, tier="B", dev=dev) for ds in fold["train"]]
    combined     = combine_graphs(train_graphs)
    test_graph   = load_graph(fold["test"], tier="B", dev=dev)

    max_feat = combined.edge_attr.shape[1]
    d        = test_graph.edge_attr.shape[1]
    if d < max_feat:
        pad = torch.zeros(test_graph.edge_attr.shape[0], max_feat - d)
        test_graph.edge_attr   = torch.cat([test_graph.edge_attr,   pad], dim=1)
        test_graph.edge_attr_q = torch.cat([test_graph.edge_attr_q, pad], dim=1)
    elif d > max_feat:
        test_graph.edge_attr   = test_graph.edge_attr[:, :max_feat]
        test_graph.edge_attr_q = test_graph.edge_attr_q[:, :max_feat]

    if no_features:
        combined.edge_attr     = torch.ones_like(combined.edge_attr)
        combined.edge_attr_q   = torch.ones_like(combined.edge_attr_q)
        test_graph.edge_attr   = torch.ones_like(test_graph.edge_attr)
        test_graph.edge_attr_q = torch.ones_like(test_graph.edge_attr_q)

    return combined, test_graph


def _load_temporal(fold: dict, dev: bool):
    """
    Load structure-only train + test graphs for temporal models.

    Context-subgraph edge attributes are set to a constant 1 scalar,
    which is the standard setting for TS-SAGE and TS-GIB.
    """
    train_graphs = []
    for ds in fold["train"]:
        g = copy.copy(load_graph(ds, tier="B", dev=dev))
        E = g.edge_attr.shape[0]
        g.edge_attr   = torch.ones(E, 1)
        g.edge_attr_q = torch.ones(E, 1)
        train_graphs.append(g)
    combined = combine_graphs(train_graphs)

    g_test = copy.copy(load_graph(fold["test"], tier="B", dev=dev))
    E = g_test.edge_attr.shape[0]
    g_test.edge_attr   = torch.ones(E, 1)
    g_test.edge_attr_q = torch.ones(E, 1)
    return combined, g_test


# ── GIB training loop ─────────────────────────────────────────────────────────

def _train_gib(
    combined, device: str, use_quantile: bool, seed: int, beta: float,
    epochs: int = 50, patience: int = 10, batch_size: int = 2048,
) -> tuple:
    """
    Training loop for GIB_EGraphSAGE.

    Uses forward_train() to obtain both logits and the KL term.
    Loss = CrossEntropy(logits, y) + beta * KL.
    Early stopping on validation MCC (minimum 10 epochs).
    Returns (trained model, best state dict).
    """
    edge_attr = combined.edge_attr_q if use_quantile else combined.edge_attr
    edge_in   = edge_attr.shape[1]
    n_edges   = edge_attr.shape[0]

    model     = GIB_EGraphSAGE(node_in=NODE_FEAT_DIM, edge_in=edge_in).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    cw        = _class_weights(combined.edge_label, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    ti, vi = train_test_split(
        np.arange(n_edges), test_size=0.2, random_state=seed,
        stratify=combined.edge_label.numpy(),
    )
    train_idx = np.array(ti, dtype=np.int64)
    val_idx   = vi.tolist()

    ea_d   = edge_attr.to(device)
    x_d    = combined.x.to(device)
    ei_d   = combined.edge_index.to(device)
    labels = combined.edge_label

    best_mcc, best_state, pat = -2.0, None, 0
    log.info(f"  Training GIB  edge_in={edge_in}  beta={beta}  epochs={epochs}")

    for epoch in range(epochs):
        model.train()
        np.random.shuffle(train_idx)

        for s in range(0, len(train_idx), batch_size):
            ids = train_idx[s:s + batch_size]
            y_b = labels[ids].to(device)
            optimizer.zero_grad()
            logits, kl = model.forward_train(x_d, ei_d[:, ids], ea_d[ids])
            loss = criterion(logits, y_b) + beta * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        result  = eval_egraphsage(model, combined, val_idx, device, use_quantile)
        val_mcc = compute_mcc(result["y_true"], result["y_pred"])
        log.info(f"  epoch {epoch + 1:02d}  val_mcc={val_mcc:.4f}")

        if val_mcc > best_mcc:
            best_mcc   = val_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat        = 0
        elif epoch >= 10:
            pat += 1
            if pat >= patience:
                log.info(f"  Early stop at epoch {epoch + 1}")
                break

    log.info(f"  Best val MCC: {best_mcc:.4f}")
    model.load_state_dict(best_state)
    return model, best_state


# ── temporal model training / evaluation ──────────────────────────────────────

def _ts_batch(model, data_list: list, q_ea: torch.Tensor, device: str):
    """Batch-forward for TS_GIB on a list of temporal subgraph Data objects."""
    batch = Batch.from_data_list([d.to(device) for d in data_list])
    ptr   = batch.ptr.to(device)
    u = torch.tensor([d.query_u for d in data_list],
                     dtype=torch.long, device=device) + ptr[:-1]
    v = torch.tensor([d.query_v for d in data_list],
                     dtype=torch.long, device=device) + ptr[:-1]
    return model(batch.x, batch.edge_index, batch.edge_attr, u, v, q_ea)


def _train_temporal(
    model, graph, device: str, seed: int, delta_us: int, beta_max: float,
    epochs: int = 20, patience: int = 5, batch_size: int = 2048,
    max_train_edges: int = 200_000, max_val_edges: int = 20_000, n_jobs: int = 4,
) -> tuple:
    """
    Training loop for TS-SAGE and TS-GIB.

    Temporal subgraphs are extracted per batch during training.
    Validation subgraphs are pre-extracted once and cached for speed.
    Returns (best state dict, p_src) where p_src is the validation attack rate
    used for calibrated threshold at test time.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    cw        = _class_weights(graph.edge_label, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    src_np    = graph.edge_index[0].numpy()
    dst_np    = graph.edge_index[1].numpy()
    time_np   = graph.edge_time.numpy()
    labels_np = graph.edge_label.numpy()
    n         = len(labels_np)

    ti_full, vi = train_test_split(
        np.arange(n), test_size=0.2, random_state=seed, stratify=labels_np,
    )
    ti_full = np.array(ti_full, dtype=np.int64)
    vi      = np.array(vi,      dtype=np.int64)

    if len(vi) > max_val_edges:
        _, vi_sub = train_test_split(
            vi, test_size=max_val_edges / len(vi),
            random_state=seed + 999, stratify=labels_np[vi],
        )
        vi = np.array(vi_sub, dtype=np.int64)

    p_src = float(labels_np[vi].mean())
    log.info(f"  train pool={len(ti_full):,}  val={len(vi):,}  p_src={p_src:.4f}")

    log.info(f"  Pre-extracting {len(vi):,} val subgraphs (n_jobs={n_jobs}) ...")
    val_cache = batch_build_subgraphs(
        src_np, dst_np, time_np,
        src_np[vi], dst_np[vi], time_np[vi],
        delta_us=delta_us, max_edges=1024,
        node_feat_dim=NODE_FEAT_DIM, seed=seed, n_jobs=n_jobs,
    )

    best_mcc, best_state, pat_cnt = -2.0, None, 0
    ep_rng      = np.random.RandomState(seed)
    WARMUP_EPOCHS = 5

    for epoch in range(epochs):
        beta = beta_max * min(1.0, (epoch + 1) / max(1, WARMUP_EPOCHS))
        model.train()

        if len(ti_full) > max_train_edges:
            ti_arr, _ = train_test_split(
                ti_full, test_size=1.0 - max_train_edges / len(ti_full),
                random_state=ep_rng.randint(0, 2 ** 31),
                stratify=labels_np[ti_full],
            )
            ti_arr = np.array(ti_arr, dtype=np.int64)
        else:
            ti_arr = ti_full.copy()
        ep_rng.shuffle(ti_arr)

        ep_loss, n_batches = 0.0, 0
        for start in range(0, len(ti_arr), batch_size):
            ids   = ti_arr[start:start + batch_size]
            yl    = torch.as_tensor(labels_np[ids], dtype=torch.long, device=device)
            q_ea  = torch.ones(len(ids), 1, device=device)

            data_list = batch_build_subgraphs(
                src_np, dst_np, time_np,
                src_np[ids], dst_np[ids], time_np[ids],
                delta_us=delta_us, max_edges=1024,
                node_feat_dim=NODE_FEAT_DIM, seed=seed, n_jobs=n_jobs,
            )

            logits, kl = _ts_batch(model, data_list, q_ea, device)
            loss = criterion(logits, yl) + beta * kl
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss   += loss.item()
            n_batches += 1

        model.eval()
        all_preds = []
        for start in range(0, len(vi), batch_size):
            ids      = vi[start:start + batch_size]
            q_ea     = torch.ones(len(ids), 1, device=device)
            dl_slice = val_cache[start:start + batch_size]
            with torch.no_grad():
                logits, _ = _ts_batch(model, dl_slice, q_ea, device)
            all_preds.append(logits.argmax(1).cpu().numpy())

        val_mcc = compute_mcc(labels_np[vi], np.concatenate(all_preds))
        log.info(f"  epoch {epoch + 1:02d}  β={beta:.4f}"
                 f"  loss={ep_loss / max(1, n_batches):.4f}  val_mcc={val_mcc:.4f}")

        if val_mcc > best_mcc:
            best_mcc   = val_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt    = 0
        elif epoch >= 3:
            pat_cnt += 1
            if pat_cnt >= patience:
                log.info(f"  Early stop at epoch {epoch + 1}")
                break

    log.info(f"  Best val MCC: {best_mcc:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    return best_state, p_src


@torch.no_grad()
def _eval_temporal(
    model, graph, device: str, delta_us: int, p_src: float,
    batch_size: int = 2048, max_eval_edges: int = 100_000, n_jobs: int = 4,
) -> dict:
    """
    Evaluate a temporal model on a test graph.

    Applies a calibrated threshold: top-k% of scores where k = p_src
    (the validation-set attack rate), which avoids assuming a 0.5 threshold.

    Returns dict with y_true, y_pred (threshold=0.5), y_pred_cal (calibrated),
    y_score, auroc, and cal_threshold.
    """
    model.eval()
    src_np    = graph.edge_index[0].numpy()
    dst_np    = graph.edge_index[1].numpy()
    time_np   = graph.edge_time.numpy()
    labels_np = graph.edge_label.numpy()
    n_total   = len(labels_np)

    if n_total > max_eval_edges:
        _, eval_idx = train_test_split(
            np.arange(n_total), test_size=max_eval_edges / n_total,
            random_state=42, stratify=labels_np,
        )
        eval_idx = np.sort(np.array(eval_idx, dtype=np.int64))
        log.info(f"  Test capped: {n_total:,} → {len(eval_idx):,}")
    else:
        eval_idx = np.arange(n_total, dtype=np.int64)

    t0 = time.time()
    all_data = batch_build_subgraphs(
        src_np, dst_np, time_np,
        src_np[eval_idx], dst_np[eval_idx], time_np[eval_idx],
        delta_us=delta_us, max_edges=1024,
        node_feat_dim=NODE_FEAT_DIM, seed=0, n_jobs=n_jobs,
    )
    log.info(f"  Test subgraph extraction: {time.time() - t0:.1f}s")

    all_preds, all_scores = [], []
    for start in range(0, len(eval_idx), batch_size):
        ids_b  = eval_idx[start:start + batch_size]
        q_ea   = torch.ones(len(ids_b), 1, device=device)
        logits, _ = _ts_batch(model, all_data[start:start + batch_size], q_ea, device)
        probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_scores.append(probs)
        all_preds.append(preds)

    y_score = np.concatenate(all_scores)
    y_pred  = np.concatenate(all_preds)
    y_true  = labels_np[eval_idx]

    k             = max(1, int(round(p_src * len(y_score))))
    cal_threshold = float(np.partition(y_score, -k)[-k])
    y_pred_cal    = (y_score >= cal_threshold).astype(int)

    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auroc = float("nan")

    return {
        "y_true":         y_true,
        "y_pred":         y_pred,
        "y_pred_cal":     y_pred_cal,
        "y_score":        y_score,
        "auroc":          auroc,
        "cal_threshold":  cal_threshold,
    }


# ── per-fold runner ───────────────────────────────────────────────────────────

def run_fold(
    model_name: str, fold: dict, seed: int, dev: bool, beta: float,
    no_features: bool, delta_secs: int, epochs: int | None, device: str,
) -> None:
    """Train and evaluate one model on one LODO fold."""
    train_dsets = fold["train"]
    test_dset   = fold["test"]

    feat_tag = "nofeat" if no_features else "raw"
    if model_name == "egraphsage":
        exp_id = f"egraphsage_{feat_tag}"
    elif model_name == "gib":
        exp_id = f"gib_b{beta}_{feat_tag}"
    elif model_name == "ts-sage":
        exp_id = f"ts-sage_d{delta_secs}"
    else:  # ts-gib
        exp_id = f"ts-gib_b{beta}_d{delta_secs}"

    if already_done(exp_id, seed, test_dset):
        log.info(f"  Skipping {exp_id} seed={seed} test={test_dset} (already in results.csv)")
        return

    log.info(f"\n{'=' * 60}")
    log.info(f"  model={model_name}  test={test_dset}  seed={seed}")
    if model_name in ("gib", "ts-gib"):
        log.info(f"  beta={beta}")
    if no_features:
        log.info(f"  features=structure-only")
    log.info(f"{'=' * 60}")

    t0         = time.time()
    is_temporal = model_name in ("ts-sage", "ts-gib")

    # ── load data ──────────────────────────────────────────────────────────────
    if is_temporal:
        combined, test_graph = _load_temporal(fold, dev)
    else:
        combined, test_graph = _load_static(fold, dev, no_features)
    use_quantile = not no_features

    # ── train ──────────────────────────────────────────────────────────────────
    if model_name == "egraphsage":
        edge_in   = (combined.edge_attr_q if use_quantile else combined.edge_attr).shape[1]
        n_edges   = combined.edge_attr.shape[0]
        ti, vi    = train_test_split(
            np.arange(n_edges), test_size=0.2, random_state=seed,
            stratify=combined.edge_label.numpy(),
        )
        val_split = {"train": ti.tolist(), "val": vi.tolist()}
        model     = EdgeAwareSAGE(node_in=NODE_FEAT_DIM, edge_in=edge_in)
        best_state = train_egraphsage(
            model, combined, val_split=val_split, device=device,
            use_quantile=use_quantile, epochs=epochs or 50,
        )
        model.load_state_dict(best_state)
        p_src = None

    elif model_name == "gib":
        model, best_state = _train_gib(
            combined, device, use_quantile, seed, beta, epochs=epochs or 50,
        )
        p_src = None

    else:  # ts-sage or ts-gib
        use_bottleneck = (model_name == "ts-gib")
        model = TS_GIB(
            node_in=NODE_FEAT_DIM, ctx_edge_in=1, q_edge_in=1,
            hidden=128, use_bottleneck=use_bottleneck, num_domains=0,
        )
        best_state, p_src = _train_temporal(
            model, combined, device, seed,
            delta_us=delta_secs * 1_000_000,
            beta_max=beta if use_bottleneck else 0.0,
            epochs=epochs or 20,
        )

    # ── evaluate ───────────────────────────────────────────────────────────────
    if is_temporal:
        result  = _eval_temporal(model, test_graph, device,
                                 delta_us=delta_secs * 1_000_000, p_src=p_src)
        metrics = compute_all_metrics(result["y_true"], result["y_pred_cal"])
        mcc_val = metrics["mcc"]
        mcc_key = "calibrated_mcc"
    else:
        result  = eval_egraphsage(model, test_graph, device=device,
                                  use_quantile=use_quantile)
        metrics = compute_all_metrics(result["y_true"], result["y_pred"])
        mcc_val = metrics["mcc"]
        mcc_key = "mcc"

    elapsed = time.time() - t0

    # ── log results ────────────────────────────────────────────────────────────
    log.info(f"\n  Results  test={test_dset}  seed={seed}:")
    log.info(f"    {mcc_key}:   {mcc_val:.4f}")
    log.info(f"    macro-F1:  {metrics['macro_f1']:.4f}")
    if is_temporal:
        log.info(f"    AUROC:     {result['auroc']:.4f}")
    log.info(f"    wall_time: {elapsed:.1f}s")

    log_result(exp_id, seed, train_dsets, test_dset, mcc_key,    mcc_val,              elapsed)
    log_result(exp_id, seed, train_dsets, test_dset, "macro_f1", metrics["macro_f1"],  elapsed)
    if is_temporal:
        log_result(exp_id, seed, train_dsets, test_dset, "auroc",   result["auroc"],   elapsed)
        log_result(exp_id, seed, train_dsets, test_dset, "p_src",   p_src,             0.0)
    save_model(exp_id, seed, test_dset, best_state)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Train and evaluate a GNN model for cross-dataset NIDS (LODO).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models
------
  egraphsage   Edge-aware GraphSAGE — fast, strong baseline with raw flow features
  gib          GIB-EGraphSAGE — adds variational bottleneck to reduce dataset bias
  ts-sage      Temporal-Subgraph SAGE — uses local temporal context per flow
  ts-gib       Temporal-Subgraph GIB — temporal context + bottleneck (best reported)

Examples
--------
  python main.py --model egraphsage
  python main.py --model egraphsage --no-features --seeds 0
  python main.py --model gib --beta 0.001 0.01 0.1
  python main.py --model ts-sage --no-dev
  python main.py --model ts-gib --beta 0.01 --seeds 0 1 2 --no-dev
  python main.py --model ts-gib --beta 0.01 --folds lycos_ids2017 unsw_nb15
        """,
    )
    parser.add_argument(
        "--model", required=True,
        choices=["egraphsage", "gib", "ts-sage", "ts-gib"],
        help="Model architecture to use.",
    )
    parser.add_argument(
        "--beta", nargs="+", type=float, default=[0.01],
        metavar="BETA",
        help="KL-divergence weight β for GIB / TS-GIB.  "
             "Pass multiple values to sweep.  (default: 0.01)",
    )
    parser.add_argument(
        "--no-features", action="store_true",
        help="Replace all edge features with 1.0 (structure-only ablation).  "
             "Not applicable to ts-sage / ts-gib, which are always structure-only.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0],
        metavar="SEED",
        help="Random seed(s) to run.  (default: 0)",
    )
    parser.add_argument(
        "--folds", nargs="+", default=None,
        metavar="DATASET",
        help="Restrict to these test datasets.  "
             "Choices: lycos_ids2017 cic_ids2018 unsw_nb15 ton_iot.  "
             "(default: all four)",
    )
    parser.add_argument(
        "--dev", action="store_true", default=True,
        help="Use dev subsamples (≤2 M rows/dataset, faster).  [default: on]",
    )
    parser.add_argument(
        "--no-dev", dest="dev", action="store_false",
        help="Use full datasets (slower, required for final numbers).",
    )
    parser.add_argument(
        "--delta", type=int, default=60, metavar="SECONDS",
        help="Temporal context window Δ in seconds for ts-sage / ts-gib.  (default: 60)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Max training epochs.  (default: 50 for static models, 20 for temporal)",
    )
    parser.add_argument(
        "--device", default=None,
        help="PyTorch device string.  (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  dev={args.dev}  |  model={args.model}")

    run_folds = ALL_FOLDS
    if args.folds:
        run_folds = [f for f in ALL_FOLDS if f["test"] in args.folds]
        if not run_folds:
            log.error(f"No LODO folds match: {args.folds}")
            sys.exit(1)

    betas = args.beta if args.model in ("gib", "ts-gib") else [args.beta[0]]

    for beta in betas:
        for seed in args.seeds:
            for fold in run_folds:
                seed_everything(seed)
                run_fold(
                    model_name  = args.model,
                    fold        = fold,
                    seed        = seed,
                    dev         = args.dev,
                    beta        = beta,
                    no_features = args.no_features,
                    delta_secs  = args.delta,
                    epochs      = args.epochs,
                    device      = device,
                )

    log.info("\nAll runs complete.  Results → results/results.csv")


if __name__ == "__main__":
    main()
