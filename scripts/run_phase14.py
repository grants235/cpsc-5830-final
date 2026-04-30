#!/usr/bin/env python3
"""
Phase 14: TS-GIB with raw (un-encoded) flow features in BOTH context and query edges.

Unlike E13.1 (quantile features on the query edge only, constant context edges),
E14 passes the original edge_attr to the temporal subgraph builder so every
context edge also carries real flow statistics.  The query edge uses the same
raw features, sharing the context encoder (q_enc = None in TS_GIB).

E14 β sweep: [10, 1, 0.1, 0.01] — high β first to see compression effects.
All 4 LODO folds, single seed.

Usage:
    python scripts/run_phase14.py                      # full sweep, all folds
    python scripts/run_phase14.py --betas 0.1 0.01     # subset of betas
    python scripts/run_phase14.py --folds cic_ids2018  # single fold
    python scripts/run_phase14.py --no-dev             # full datasets
"""

import argparse
import copy
import csv
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split as _tts

from run_phase4 import ALL_FOLDS
from run_phase12 import (
    _graph_arrays, _delta_us,
    _ts_gib_forward_batch, _calibrated_mcc, _oracle_mcc,
    _all_calibration_mccs,
    DELTA_SECS, BETA_WARMUP, MAX_EPOCHS, PATIENCE,
    BATCH_SIZE, MAX_TRAIN_EDGES, MAX_VAL_EDGES, MAX_EVAL_EDGES,
    MAX_SUB_EDGES, NODE_FEAT_DIM, N_JOBS,
)

from src.utils.logging import setup_logging, log_result, already_done, save_model, MODELS_DIR
from src.utils.seeding import seed_everything
from src.utils.metrics import compute_all_metrics, compute_mcc
from src.data.graph_builder import load_graph, combine_graphs
from src.data.temporal_subgraph import batch_build_subgraphs
from src.models.temporal_gnn import TS_GIB
from src.train.train_loops import _class_weights

log = logging.getLogger(__name__)

BETA_SWEEP = [10.0, 1.0, 0.1, 0.01]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_fold_raw(fold: dict, dev: bool):
    """
    Load fold with raw (un-encoded) edge features.
    Returns (combined, test_graph, feat_dim).
    Aligns test edge_attr to training feat_dim by padding or cropping.
    """
    train_graphs = [load_graph(ds, tier="B", dev=dev) for ds in fold["train"]]
    combined     = combine_graphs(train_graphs)
    test_graph   = load_graph(fold["test"], tier="B", dev=dev)

    feat_dim = combined.edge_attr.shape[1]
    d = test_graph.edge_attr.shape[1]
    if d < feat_dim:
        test_graph.edge_attr = torch.cat(
            [test_graph.edge_attr,
             torch.zeros(test_graph.edge_attr.shape[0], feat_dim - d)], dim=1)
    elif d > feat_dim:
        test_graph.edge_attr = test_graph.edge_attr[:, :feat_dim]

    return combined, test_graph, feat_dim


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _train_e14(
    model: TS_GIB,
    graph,
    feat_np: np.ndarray,        # [E_combined, feat_dim] raw edge_attr
    device: str,
    seed: int,
    exp_id: str,
    test_dset: str,
    delta_us: int,
    beta_max: float,
    epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    max_train_edges: int = MAX_TRAIN_EDGES,
    max_val_edges: int = MAX_VAL_EDGES,
    warmup_epochs: int = BETA_WARMUP,
    n_jobs: int = N_JOBS,
) -> tuple:
    """Train TS_GIB with raw features in context and query. Returns (best_state, p_src)."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    cw        = _class_weights(graph.edge_label, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    src_np, dst_np, time_np = _graph_arrays(graph)
    labels_np = graph.edge_label.numpy()
    n         = len(labels_np)

    _, vi = _tts(np.arange(n), test_size=0.2, random_state=seed, stratify=labels_np)
    ti_full = np.setdiff1d(np.arange(n), vi)
    vi = np.array(vi, dtype=np.int64)

    if len(vi) > max_val_edges:
        _, vi_sub = _tts(vi, test_size=max_val_edges / len(vi),
                         random_state=seed + 999, stratify=labels_np[vi])
        vi = np.sort(np.array(vi_sub, dtype=np.int64))

    p_src = float(labels_np[vi].mean())
    log.info(f"  train={len(ti_full):,}  val={len(vi):,}  p_src={p_src:.4f}")

    # Pre-cache val subgraphs with raw features
    log.info(f"  Pre-extracting {len(vi):,} val subgraphs …")
    t_c = time.time()
    val_cache = batch_build_subgraphs(
        src_np, dst_np, time_np,
        src_np[vi], dst_np[vi], time_np[vi],
        delta_us=delta_us, max_edges=MAX_SUB_EDGES,
        node_feat_dim=NODE_FEAT_DIM, seed=seed, n_jobs=n_jobs,
        edge_attr_np=feat_np,
    )
    log.info(f"  Val cache ready in {time.time()-t_c:.1f}s")

    best_mcc, best_state, pat_cnt = -2.0, None, 0
    ep_rng = np.random.RandomState(seed)

    for epoch in range(epochs):
        beta = beta_max * min(1.0, (epoch + 1) / max(1, warmup_epochs))
        model.train()

        if len(ti_full) > max_train_edges:
            _, ti_arr = _tts(ti_full, test_size=max_train_edges / len(ti_full),
                             random_state=ep_rng.randint(0, 2**31),
                             stratify=labels_np[ti_full])
            ti_arr = np.array(ti_arr, dtype=np.int64)
        else:
            ti_arr = ti_full.copy()
        ep_rng.shuffle(ti_arr)

        ep_loss, n_batches = 0.0, 0
        for start in range(0, len(ti_arr), batch_size):
            ids  = ti_arr[start:start + batch_size]
            yl   = torch.as_tensor(labels_np[ids], dtype=torch.long, device=device)
            # Query features: raw edge_attr for this batch
            q_ea = torch.as_tensor(feat_np[ids], dtype=torch.float32, device=device)
            data_list = batch_build_subgraphs(
                src_np, dst_np, time_np,
                src_np[ids], dst_np[ids], time_np[ids],
                delta_us=delta_us, max_edges=MAX_SUB_EDGES,
                node_feat_dim=NODE_FEAT_DIM, seed=seed, n_jobs=n_jobs,
                edge_attr_np=feat_np,
            )
            logits, kl = _ts_gib_forward_batch(model, data_list, q_ea, device)
            loss = criterion(logits, yl) + beta * kl
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item(); n_batches += 1

        # Validation (uses pre-cached subgraphs)
        model.eval()
        all_preds = []
        for start in range(0, len(vi), batch_size):
            ids  = vi[start:start + batch_size]
            q_ea = torch.as_tensor(feat_np[ids], dtype=torch.float32, device=device)
            dl   = val_cache[start:start + batch_size]
            with torch.no_grad():
                logits, _ = _ts_gib_forward_batch(model, dl, q_ea, device)
            all_preds.append(logits.argmax(1).cpu().numpy())

        val_mcc = compute_mcc(labels_np[vi], np.concatenate(all_preds))
        log.info(f"  epoch {epoch+1:02d}  β={beta:.4f}"
                  f"  loss={ep_loss/max(1,n_batches):.4f}  val_mcc={val_mcc:.4f}")

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
    return best_state, p_src


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_e14(
    model: TS_GIB,
    test_graph,
    feat_np_test: np.ndarray,   # [E_test, feat_dim] raw edge_attr
    device: str,
    delta_us: int,
    p_src: float,
    batch_size: int = BATCH_SIZE,
    max_eval_edges: int = MAX_EVAL_EDGES,
    n_jobs: int = N_JOBS,
) -> dict:
    model.eval()
    src_np, dst_np, time_np = _graph_arrays(test_graph)
    labels_np = test_graph.edge_label.numpy()
    n_total   = len(labels_np)

    if n_total > max_eval_edges:
        _, eval_idx = _tts(np.arange(n_total),
                           test_size=max_eval_edges / n_total,
                           random_state=42, stratify=labels_np)
        eval_idx = np.sort(np.array(eval_idx, dtype=np.int64))
        log.info(f"  Test capped: {n_total:,} → {len(eval_idx):,}")
    else:
        eval_idx = np.arange(n_total, dtype=np.int64)

    labels_eval = labels_np[eval_idx]

    t0 = time.time()
    all_data = batch_build_subgraphs(
        src_np, dst_np, time_np,
        src_np[eval_idx], dst_np[eval_idx], time_np[eval_idx],
        delta_us=delta_us, max_edges=MAX_SUB_EDGES,
        node_feat_dim=NODE_FEAT_DIM, seed=0, n_jobs=n_jobs,
        edge_attr_np=feat_np_test,
    )
    log.info(f"  Test subgraph extraction: {time.time()-t0:.1f}s")

    all_scores, all_preds = [], []
    for start in range(0, len(eval_idx), batch_size):
        ids_b  = eval_idx[start:start + batch_size]
        q_ea   = torch.as_tensor(feat_np_test[ids_b], dtype=torch.float32, device=device)
        dl     = all_data[start:start + batch_size]
        logits, _ = _ts_gib_forward_batch(model, dl, q_ea, device)
        all_scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        all_preds.append(logits.argmax(1).cpu().numpy())

    y_score = np.concatenate(all_scores)
    y_pred  = np.concatenate(all_preds)
    cal_mcc, _ = _calibrated_mcc(y_score, labels_eval, p_src)
    orc_mcc, _ = _oracle_mcc(y_score, labels_eval)
    try:
        auroc = float(roc_auc_score(labels_eval, y_score))
    except Exception:
        auroc = float("nan")
    metrics = compute_all_metrics(labels_eval, y_pred)
    all_cal = _all_calibration_mccs(y_score, labels_eval, p_src)
    return {
        "y_true": labels_eval, "y_pred": y_pred, "y_score": y_score,
        "reported_mcc": compute_mcc(labels_eval, y_pred),
        "calibrated_mcc": cal_mcc, "oracle_mcc": orc_mcc,
        "auroc": auroc, "auprc": metrics.get("auprc", float("nan")),
        "macro_f1": metrics["macro_f1"], "p_src": p_src,
        "all_cal": all_cal,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_e14(betas=None, seeds=None, dev=True, folds=None):
    betas  = betas or BETA_SWEEP
    seeds  = seeds or [0]
    folds  = folds or ALL_FOLDS
    du     = _delta_us(DELTA_SECS)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"=== E14  TS-GIB raw context+query  β sweep={betas}  seeds={seeds} ===")

    for beta in betas:
        exp_id = f"E14_ts_gib_rawctx_b{beta}"
        log.info(f"\n{'='*60}")
        log.info(f"  β = {beta}  exp_id = {exp_id}")

        for seed in seeds:
            for fold in folds:
                train_dsets = fold["train"]
                test_dset   = fold["test"]

                if already_done(exp_id, seed, test_dset):
                    log.info(f"  Skip {exp_id} seed={seed} test={test_dset}")
                    continue

                seed_everything(seed)
                t0 = time.time()
                log.info(f"\n  seed={seed}  test={test_dset}  β={beta}")

                combined, test_graph, feat_dim = _load_fold_raw(fold, dev)
                feat_np_train = combined.edge_attr.numpy().astype(np.float32)
                feat_np_test  = test_graph.edge_attr.numpy().astype(np.float32)
                log.info(f"  feat_dim={feat_dim}  "
                          f"train_edges={len(feat_np_train):,}  "
                          f"test_edges={len(feat_np_test):,}")

                # ctx_edge_in == q_edge_in → shared encoder (q_enc = None)
                model = TS_GIB(
                    node_in=NODE_FEAT_DIM,
                    ctx_edge_in=feat_dim,
                    q_edge_in=feat_dim,
                    hidden=128,
                    use_bottleneck=True,
                    num_domains=0,
                )

                best_state, p_src = _train_e14(
                    model, combined, feat_np_train, device,
                    seed, exp_id, test_dset, du, beta_max=beta,
                )

                model.eval()
                result  = _eval_e14(model, test_graph, feat_np_test, device, du, p_src)
                elapsed = time.time() - t0

                log.info(f"  [{exp_id}] seed={seed} fold={test_dset}"
                          f"  rep={result['reported_mcc']:.4f}"
                          f"  cal={result['calibrated_mcc']:.4f}"
                          f"  orc={result['oracle_mcc']:.4f}"
                          f"  auroc={result['auroc']:.4f}"
                          f"  ({elapsed/60:.1f} min)")

                log_result(exp_id, seed, train_dsets, test_dset, "mcc",
                           result["reported_mcc"], elapsed)
                log_result(exp_id, seed, train_dsets, test_dset, "calibrated_mcc",
                           result["calibrated_mcc"], elapsed)
                log_result(exp_id, seed, train_dsets, test_dset, "oracle_mcc",
                           result["oracle_mcc"], elapsed)
                log_result(exp_id, seed, train_dsets, test_dset, "auroc",
                           result["auroc"], elapsed)
                log_result(exp_id, seed, train_dsets, test_dset, "macro_f1",
                           result["macro_f1"], elapsed)
                log_result(exp_id, seed, train_dsets, test_dset, "p_src",
                           p_src, 0.0)

                del model, combined, test_graph
                import gc; gc.collect()

        # Per-β summary
        _print_beta_summary(exp_id, seeds)


def _print_beta_summary(exp_id: str, seeds: list):
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return
    fold_cal: dict = {}
    fold_auroc: dict = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["experiment_id"] != exp_id or int(row["seed"]) not in seeds:
                continue
            td = row["test_dataset"]
            if row["metric"] == "calibrated_mcc":
                fold_cal.setdefault(td, []).append(float(row["value"]))
            elif row["metric"] == "auroc":
                fold_auroc.setdefault(td, []).append(float(row["value"]))
    if not fold_cal:
        return
    log.info(f"\n  {exp_id} summary:")
    log.info(f"  {'fold':<22} {'cal_mcc':>9} {'auroc':>9}")
    for td in sorted(fold_cal):
        cal   = np.mean(fold_cal[td])
        auroc = np.mean(fold_auroc.get(td, [float("nan")]))
        log.info(f"  {td:<22} {cal:>9.4f} {auroc:>9.4f}")
    mean_cal   = np.mean([np.mean(v) for v in fold_cal.values()])
    mean_auroc = np.mean([np.mean(v) for v in fold_auroc.values()])
    log.info(f"  {'mean':<22} {mean_cal:>9.4f} {mean_auroc:>9.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Final comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(seeds=None):
    """Compare E12.1 (no-feat), E13.1 (quant query), E14 (raw ctx+query) side by side."""
    seeds = seeds or [0]
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return

    ROWS = [
        ("E12.1 struct-only b0.01",     "E12.1_ts_gib_b0.01"),
        ("E13.1 quant-query b0.01",     "E13.1_ts_gib_raw_b0.01"),
        ("E14 raw-ctx b10",             "E14_ts_gib_rawctx_b10.0"),
        ("E14 raw-ctx b1",              "E14_ts_gib_rawctx_b1.0"),
        ("E14 raw-ctx b0.1",            "E14_ts_gib_rawctx_b0.1"),
        ("E14 raw-ctx b0.01",           "E14_ts_gib_rawctx_b0.01"),
    ]
    FOLDS = ["lycos_ids2017", "cic_ids2018", "unsw_nb15", "ton_iot"]

    data: dict = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if int(row["seed"]) not in seeds:
                continue
            for _, eid in ROWS:
                if row["experiment_id"] == eid and row["metric"] == "calibrated_mcc":
                    data.setdefault(eid, {})[row["test_dataset"]] = float(row["value"])

    log.info("\n  Calibrated MCC comparison (raw vs struct-only vs quant-query):")
    log.info(f"  {'Method':<28} {'lycos':>8} {'cic18':>8} {'unsw':>8} {'ton':>8} {'mean':>8}")
    for label, eid in ROWS:
        vals = []
        row_data = data.get(eid, {})
        for td in FOLDS:
            v = row_data.get(td, float("nan"))
            vals.append(f"{v:>8.3f}" if not np.isnan(v) else "     n/a")
        fold_means = [row_data[td] for td in FOLDS if td in row_data and not np.isnan(row_data.get(td, float("nan")))]
        mean_str = f"{np.mean(fold_means):>8.3f}" if fold_means else "     n/a"
        log.info(f"  {label:<28}" + "".join(vals) + mean_str)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Phase 14: TS-GIB raw context+query features")
    parser.add_argument("--betas",  nargs="+", type=float, default=None,
                        help="β values to sweep (default: 10 1 0.1 0.01)")
    parser.add_argument("--seeds",  nargs="+", type=int, default=None)
    parser.add_argument("--folds",  nargs="+", default=None,
                        help="Restrict to these test datasets")
    parser.add_argument("--no-dev", dest="dev", action="store_false")
    parser.add_argument("--table",  action="store_true",
                        help="Print comparison table and exit")
    args = parser.parse_args()

    if args.table:
        print_comparison_table(seeds=args.seeds or [0])
        return

    run_folds = ALL_FOLDS
    if args.folds:
        run_folds = [f for f in ALL_FOLDS if f["test"] in args.folds]
        if not run_folds:
            log.error(f"No folds match {args.folds}"); return

    run_e14(
        betas=args.betas,
        seeds=args.seeds or [0],
        dev=args.dev,
        folds=run_folds,
    )


if __name__ == "__main__":
    main()
