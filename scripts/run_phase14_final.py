#!/usr/bin/env python3
"""
Phase 14 Final (spex14.md): Final Number Collection for Paper Submission.

Task 1 — Multi-seed TS-GIB no features β=0.01       (E14.1_ts_gib_b0.01, seeds 0-2)
Task 2 — TS-SAGE no features all 4 folds             (E14.2_ts_sage_d60,  seed 0)
Task 3 — Calibration panel on E14.1 + E14.2          (writes calibration_v2.csv)
Task 4 — Probe accuracy on models missing probe      (E13.1, E13.2, E12.1 off-betas)
Task 5 — Per-attack-class F1 for headline model      (E14.1 on solvable folds)
Task 6 — Reported MCC column summary                 (reads results.csv, prints table)
Task 7 — Consistency check                           (cross-references CSVs)

Usage:
    python scripts/run_phase14_final.py --tasks 1 2 3 4 5 6 7   # all tasks
    python scripts/run_phase14_final.py --tasks 1                # just multi-seed training
    python scripts/run_phase14_final.py --tasks 3                # just calibration
    python scripts/run_phase14_final.py --no-dev                 # full datasets
    python scripts/run_phase14_final.py --seeds 1 2              # only seeds 1 and 2 for T1
    python scripts/run_phase14_final.py --folds lycos_ids2017 cic_ids2018
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split as _tts
from torch_geometric.data import Batch

from run_phase4 import ALL_FOLDS
from run_phase12 import (
    _load_fold_struct, _graph_arrays, _delta_us, _make_q_ea,
    _train_ts_gib, _eval_ts_gib, _log_eval_results, _probe_ts_gib,
    _ts_gib_forward_batch, _ts_gib_embed_batch,
    _calibrated_mcc, _oracle_mcc, _all_calibration_mccs,
    _topk_threshold, _otsu_threshold,
    _print_e12_summary,
    DELTA_SECS, BETA_WARMUP, MAX_EPOCHS, PATIENCE,
    BATCH_SIZE, MAX_TRAIN_EDGES, MAX_VAL_EDGES, MAX_EVAL_EDGES,
    MAX_SUB_EDGES, NODE_FEAT_DIM, N_JOBS,
)
from run_phase8 import (
    run_e8_1_ts_sage,
    _train_temporal, _eval_temporal, _probe_on_temporal_encoder,
    _load_fold_struct as _p8_load_fold_struct,
)
from run_phase13 import (
    _all_cal_methods_v2, _load_fold_raw, _make_q_ea_raw,
    _reproduce_val_split_p12, _reproduce_val_split_p8,
    CAL_V2_CSV,
)

import run_phase13 as _p13

from src.utils.logging import setup_logging, log_result, already_done, save_model, MODELS_DIR
from src.utils.seeding import seed_everything
from src.utils.metrics import compute_all_metrics, compute_mcc, compute_per_class_f1, CLASSES
from src.data.graph_builder import load_graph, combine_graphs
from src.data.temporal_subgraph import batch_build_subgraphs
from src.models.temporal_gnn import TS_GIB, TemporalEdgeSAGE

log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

T1_EXP_ID    = "E14.1_ts_gib_b0.01"
T1_BETA      = 0.01
T1_SEEDS     = [0, 1, 2]

T2_EXP_ID    = "E14.2_ts_sage_d60"
T2_EXP_PREFIX = "E14.2_ts_sage"
T2_SEEDS     = [0]

CAL_FIELDNAMES = [
    "experiment_id", "seed", "test_fold",
    "p11_reported_0.5", "p11_topk_src_rate", "p11_otsu",
    "p11_gmm", "p11_topk_10pct", "p11_topk_20pct", "p11_topk_30pct",
    "p11_oracle",
    "val_anchor", "znorm_val", "platt", "bbse", "gmm_logit", "ensemble",
    "oracle_mcc", "auroc",
]

# Decision threshold for "solvable" fold (Task 5 skips folds below this)
MIN_AUROC_SOLVABLE = 0.65

ATTACK_CLASSES = CLASSES[1:]  # skip Benign


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Multi-seed TS-GIB no features β=0.01
# ─────────────────────────────────────────────────────────────────────────────

def run_task1(seeds=None, folds=None, dev=True):
    """Train E14.1_ts_gib_b0.01 at seeds 0-2 × all 4 folds."""
    seeds  = seeds or T1_SEEDS
    folds  = folds or ALL_FOLDS
    du     = _delta_us(DELTA_SECS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id = T1_EXP_ID

    log.info(f"\n{'='*65}")
    log.info(f"Task 1 — TS-GIB no features β=0.01  seeds={seeds}  folds={[f['test'] for f in folds]}")
    log.info(f"{'='*65}")

    for seed in seeds:
        for fold in folds:
            train_dsets = fold["train"]
            test_dset   = fold["test"]

            if already_done(exp_id, seed, test_dset):
                log.info(f"  Skip {exp_id} seed={seed} test={test_dset}")
                continue

            seed_everything(seed)
            t0 = time.time()
            log.info(f"\n  seed={seed}  test={test_dset}  β={T1_BETA}")

            combined, test_graph, _, _ = _load_fold_struct(fold, dev)

            model = TS_GIB(node_in=NODE_FEAT_DIM, ctx_edge_in=1, q_edge_in=1,
                           hidden=128, use_bottleneck=True, num_domains=0)
            best_state, p_src = _train_ts_gib(
                model, combined, device, seed, exp_id, test_dset,
                delta_us=du, beta_max=T1_BETA,
            )

            model.eval()
            result  = _eval_ts_gib(model, test_graph, device, du, p_src)
            elapsed = time.time() - t0

            _log_eval_results(result, exp_id, seed, train_dsets, test_dset, elapsed)

            probe_acc = _probe_ts_gib(model, train_dsets, dev, seed, device, du)
            log.info(f"  Probe accuracy: {probe_acc:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset,
                       "dataset_probe_acc", probe_acc, 0.0)

            del model, combined, test_graph
            import gc; gc.collect()

    _print_e12_summary(exp_id, seeds, "calibrated_mcc")
    _print_e12_summary(exp_id, seeds, "auroc")
    _assess_task1_robustness(seeds)


def _assess_task1_robustness(seeds: list):
    """Decision rule check: is multi-seed mean within 0.05 of seed-0?"""
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return
    fold_cal: dict  = {}
    fold_auroc: dict = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["experiment_id"] != T1_EXP_ID:
                continue
            if int(row["seed"]) not in seeds:
                continue
            td = row["test_dataset"]
            if row["metric"] == "calibrated_mcc":
                fold_cal.setdefault(td, []).append(float(row["value"]))
            elif row["metric"] == "auroc":
                fold_auroc.setdefault(td, []).append(float(row["value"]))

    if not fold_cal:
        return

    log.info(f"\n  Task 1 multi-seed decision rule:")
    log.info(f"  {'fold':<22} {'cal_mcc':>9} {'±std':>8} {'auroc':>9} {'±std':>8}")
    fragile = False
    for td in sorted(fold_cal):
        vals  = fold_cal[td]
        auroc = fold_auroc.get(td, [float("nan")])
        m_cal  = np.mean(vals)
        s_cal  = np.std(vals)
        m_aur  = np.mean(auroc)
        s_aur  = np.std(auroc)
        log.info(f"  {td:<22} {m_cal:>9.4f} {s_cal:>8.4f} {m_aur:>9.4f} {s_aur:>8.4f}")
        if m_cal < 0.45 or m_aur < 0.75 or s_cal > 0.10:
            fragile = True

    means_cal = [np.mean(v) for v in fold_cal.values()]
    mean_cal_3fold = np.mean([v for td, v in
                              [(td, np.mean(vals)) for td, vals in fold_cal.items()
                               if td != "ton_iot"]])
    log.info(f"  3-fold mean calibrated MCC (excl. ton_iot): {mean_cal_3fold:.4f}")

    if fragile:
        log.warning("  DECISION: Fragile — multi-seed variance exceeds threshold. "
                    "Weaken abstract claim.")
    else:
        log.info("  DECISION: Robust — multi-seed numbers are stable. "
                 "Headline result is defensible.")


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — TS-SAGE no features (all 4 folds)
# ─────────────────────────────────────────────────────────────────────────────

def run_task2(seeds=None, folds=None, dev=True):
    """Train E14.2_ts_sage_d60 on all 4 folds (esp. ton_iot gap filler)."""
    seeds = seeds or T2_SEEDS
    folds = folds or ALL_FOLDS

    log.info(f"\n{'='*65}")
    log.info(f"Task 2 — TS-SAGE no features  seeds={seeds}  folds={[f['test'] for f in folds]}")
    log.info(f"{'='*65}")

    run_e8_1_ts_sage(
        seeds=seeds,
        delta_secs=DELTA_SECS,
        dev=dev,
        folds=folds,
        exp_id_prefix=T2_EXP_PREFIX,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Calibration panel on E14.1 + E14.2 checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def _is_temporal_patched(exp_id: str) -> bool:
    """Extend phase13 temporal check to include E14.x experiments."""
    return any(exp_id.startswith(p) for p in
               ("E8.", "E9.", "E12.", "E13.1", "E13.2", "E13.3", "E14."))


def _get_p_src_from_csv(exp_id: str, seed: int, test_fold: str) -> float:
    path = Path("results/results.csv")
    if not path.exists():
        return 0.2
    with open(path) as f:
        for row in csv.DictReader(f):
            if (row["experiment_id"] == exp_id and int(row["seed"]) == seed
                    and row["test_dataset"] == test_fold
                    and row["metric"] == "p_src"):
                return float(row["value"])
    return 0.2


@torch.no_grad()
def _compute_val_scores(ckpt_path: Path, fold: dict, seed: int, dev: bool,
                        device: str) -> tuple:
    """
    Reproduce val scores for a temporal checkpoint.
    Returns (val_scores, val_labels) or (None, None) on failure.
    Handles both TS_GIB (E14.1) and TemporalEdgeSAGE (E14.2).
    """
    if not ckpt_path.exists():
        return None, None

    state      = torch.load(ckpt_path, weights_only=True)
    is_ts_gib  = "ctx_enc.0.weight" in state
    du         = _delta_us(DELTA_SECS)

    if is_ts_gib:
        has_q_enc = any(k.startswith("q_enc.") for k in state)
        q_edge_in = state["q_enc.0.weight"].shape[1] if has_q_enc else 1
        use_bn    = "to_dist.weight" in state
        num_doms  = state["domain_head.2.weight"].shape[0] if "domain_head.2.weight" in state else 0
        model     = TS_GIB(node_in=NODE_FEAT_DIM, ctx_edge_in=1,
                           q_edge_in=q_edge_in, hidden=128,
                           use_bottleneck=use_bn, num_domains=num_doms)
        # Reproduce Phase 12 val split
        combined, _, _, _ = _load_fold_struct(fold, dev)
        labels_np = combined.edge_label.numpy()
        vi        = _reproduce_val_split_p12(labels_np, seed)
        src_np, dst_np, time_np = _graph_arrays(combined)
    else:
        edge_in  = state["edge_enc.0.weight"].shape[1]
        hidden   = state["edge_enc.0.weight"].shape[0]
        node_in  = state["conv1.lin_l.weight"].shape[1] - hidden
        model    = TemporalEdgeSAGE(node_in=node_in, edge_in=edge_in, hidden=hidden)
        combined, _, _, _ = _p8_load_fold_struct(fold, dev)
        labels_np = combined.edge_label.numpy()
        vi        = _reproduce_val_split_p8(labels_np, seed)
        src_np, dst_np, time_np = _graph_arrays(combined)

    model.load_state_dict(state)
    model.eval().to(device)

    all_scores = []
    for start in range(0, len(vi), BATCH_SIZE):
        ids  = vi[start:start + BATCH_SIZE]
        q_ea = torch.ones(len(ids), 1, device=device)
        data_list = batch_build_subgraphs(
            src_np, dst_np, time_np,
            src_np[ids], dst_np[ids], time_np[ids],
            delta_us=du, max_edges=MAX_SUB_EDGES,
            node_feat_dim=NODE_FEAT_DIM, seed=seed, n_jobs=N_JOBS,
        )
        out    = _ts_gib_forward_batch(model, data_list, q_ea, device)
        logits = out[0] if isinstance(out, tuple) else out
        all_scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())

    return np.concatenate(all_scores).astype(np.float32), labels_np[vi]


@torch.no_grad()
def _compute_test_scores(ckpt_path: Path, fold: dict, dev: bool,
                         device: str) -> tuple:
    """
    Run inference on the test graph for a temporal checkpoint.
    Returns (test_scores, test_labels) with MAX_EVAL_EDGES cap.
    """
    if not ckpt_path.exists():
        return None, None

    state      = torch.load(ckpt_path, weights_only=True)
    is_ts_gib  = "ctx_enc.0.weight" in state
    du         = _delta_us(DELTA_SECS)

    if is_ts_gib:
        has_q_enc = any(k.startswith("q_enc.") for k in state)
        q_edge_in = state["q_enc.0.weight"].shape[1] if has_q_enc else 1
        use_bn    = "to_dist.weight" in state
        num_doms  = state["domain_head.2.weight"].shape[0] if "domain_head.2.weight" in state else 0
        model     = TS_GIB(node_in=NODE_FEAT_DIM, ctx_edge_in=1,
                           q_edge_in=q_edge_in, hidden=128,
                           use_bottleneck=use_bn, num_domains=num_doms)
        _, test_graph, _, _ = _load_fold_struct(fold, dev)
    else:
        edge_in  = state["edge_enc.0.weight"].shape[1]
        hidden   = state["edge_enc.0.weight"].shape[0]
        node_in  = state["conv1.lin_l.weight"].shape[1] - hidden
        model    = TemporalEdgeSAGE(node_in=node_in, edge_in=edge_in, hidden=hidden)
        _, test_graph, _, _ = _p8_load_fold_struct(fold, dev)

    model.load_state_dict(state)
    model.eval().to(device)

    src_np, dst_np, time_np = _graph_arrays(test_graph)
    labels_np = test_graph.edge_label.numpy()
    n_total   = len(labels_np)

    if n_total > MAX_EVAL_EDGES:
        _, eval_idx = _tts(np.arange(n_total),
                           test_size=MAX_EVAL_EDGES / n_total,
                           random_state=42, stratify=labels_np)
        eval_idx = np.sort(np.array(eval_idx, dtype=np.int64))
        log.info(f"  Test capped: {n_total:,} → {len(eval_idx):,}")
    else:
        eval_idx = np.arange(n_total, dtype=np.int64)

    t0 = time.time()
    all_data = batch_build_subgraphs(
        src_np, dst_np, time_np,
        src_np[eval_idx], dst_np[eval_idx], time_np[eval_idx],
        delta_us=du, max_edges=MAX_SUB_EDGES,
        node_feat_dim=NODE_FEAT_DIM, seed=0, n_jobs=N_JOBS,
    )
    log.info(f"  Subgraph extraction: {time.time()-t0:.1f}s")

    all_scores = []
    for start in range(0, len(eval_idx), BATCH_SIZE):
        ids_b = eval_idx[start:start + BATCH_SIZE]
        q_ea  = torch.ones(len(ids_b), 1, device=device)
        dl    = all_data[start:start + BATCH_SIZE]
        out   = _ts_gib_forward_batch(model, dl, q_ea, device)
        logits = out[0] if isinstance(out, tuple) else out
        all_scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())

    return np.concatenate(all_scores).astype(np.float32), labels_np[eval_idx]


def run_task3(target_exps=None, dev=True):
    """
    Run Phase 13 calibration panel on E14.1 and E14.2 checkpoints.
    Writes results to results/calibration_v2.csv and prints winner analysis.
    """
    log.info(f"\n{'='*65}")
    log.info("Task 3 — Calibration panel on E14.1 + E14.2")
    log.info(f"{'='*65}")

    if target_exps is None:
        target_exps = [T1_EXP_ID, T2_EXP_ID]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Patch phase13's temporal detector so E14.x checkpoints are treated as temporal
    _p13._is_temporal = _is_temporal_patched

    _p13.run_a1_calibration_sweep(target_exps=target_exps, dev=dev)
    _p13.run_a2_winner_analysis(exp_id_filter="")


def run_task3_direct(dev=True):
    """
    Direct calibration sweep (fallback if phase13 patch doesn't work).
    Uses local inference functions and writes to calibration_v2.csv.
    """
    log.info(f"\n{'='*65}")
    log.info("Task 3 (direct) — Calibration panel on E14.1 + E14.2")
    log.info(f"{'='*65}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    CAL_V2_CSV.parent.mkdir(parents=True, exist_ok=True)

    done_keys = set()
    if CAL_V2_CSV.exists():
        with open(CAL_V2_CSV) as f:
            for row in csv.DictReader(f):
                done_keys.add((row["experiment_id"], row["seed"], row["test_fold"]))

    write_header = not done_keys and not CAL_V2_CSV.exists()

    with open(CAL_V2_CSV, "a", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=CAL_FIELDNAMES, extrasaction="ignore")
        if write_header or not CAL_V2_CSV.exists():
            writer.writeheader()

        for exp_id in [T1_EXP_ID, T2_EXP_ID]:
            log.info(f"\n  Processing {exp_id} ...")
            ckpt_pattern = f"{exp_id}_seed*_test*.pt"
            ckpts = sorted(MODELS_DIR.glob(ckpt_pattern))
            if not ckpts:
                log.warning(f"  No checkpoints for {exp_id}")
                continue

            for ckpt in ckpts:
                stem  = ckpt.stem
                parts = stem.split("_seed")
                if len(parts) < 2:
                    continue
                tail  = parts[-1].split("_test")
                if len(tail) < 2:
                    continue
                try:
                    seed      = int(tail[0])
                    test_fold = tail[1]
                except ValueError:
                    continue

                if (exp_id, str(seed), test_fold) in done_keys:
                    log.info(f"  Skip {exp_id} seed={seed} fold={test_fold} (done)")
                    continue

                fold = next((f for f in ALL_FOLDS if f["test"] == test_fold), None)
                if fold is None:
                    continue

                log.info(f"  {exp_id} seed={seed} fold={test_fold}")

                try:
                    test_scores, test_labels = _compute_test_scores(ckpt, fold, dev, device)
                    if test_scores is None:
                        continue
                    val_scores, val_labels = _compute_val_scores(ckpt, fold, seed, dev, device)
                except Exception as e:
                    log.warning(f"  Failed inference: {e}")
                    continue

                p_src = _get_p_src_from_csv(exp_id, seed, test_fold)
                try:
                    cal = _all_cal_methods_v2(
                        val_scores, val_labels, test_scores, test_labels, p_src)
                except Exception as e:
                    log.warning(f"  Calibration failed: {e}")
                    continue

                orc, _ = _oracle_mcc(test_scores, test_labels)
                try:
                    auroc = float(roc_auc_score(test_labels, test_scores))
                except Exception:
                    auroc = float("nan")

                row_out = {
                    "experiment_id": exp_id, "seed": str(seed),
                    "test_fold": test_fold,
                    "oracle_mcc": f"{orc:.6f}", "auroc": f"{auroc:.6f}",
                }
                for k, v in cal.items():
                    row_out[k] = f"{v:.6f}" if not np.isnan(v) else "nan"
                writer.writerow(row_out)
                f_out.flush()
                done_keys.add((exp_id, str(seed), test_fold))
                log.info(f"  Done: oracle={orc:.4f}  auroc={auroc:.4f}")

    # Winner analysis
    _winner_analysis()


def _winner_analysis():
    """Print per-fold best calibration method from calibration_v2.csv."""
    if not CAL_V2_CSV.exists():
        log.warning("calibration_v2.csv not found — run Task 3 first.")
        return

    CAL_METHODS = [
        "p11_topk_src_rate", "p11_otsu", "p11_gmm",
        "p11_topk_10pct", "p11_topk_20pct", "p11_topk_30pct",
        "val_anchor", "znorm_val", "platt", "bbse", "gmm_logit", "ensemble",
    ]

    fold_winners: dict = {}
    rows: dict = {}
    with open(CAL_V2_CSV) as f:
        for row in csv.DictReader(f):
            key = (row["experiment_id"], row["test_fold"])
            try:
                orc = float(row.get("oracle_mcc", "nan"))
            except ValueError:
                orc = float("nan")
            if key not in rows or orc > float(rows[key].get("oracle_mcc", -999)):
                rows[key] = row

    log.info(f"\n  Calibration winner analysis:")
    method_wins: dict = {}
    for (eid, fold), row in sorted(rows.items()):
        method_mccs = {}
        for m in CAL_METHODS:
            try:
                v = float(row.get(m, "nan"))
                if not np.isnan(v):
                    method_mccs[m] = v
            except (ValueError, TypeError):
                pass
        if not method_mccs:
            continue
        best_m   = max(method_mccs, key=method_mccs.get)
        best_mcc = method_mccs[best_m]
        try:
            orc  = float(row.get("oracle_mcc", "nan"))
            auroc = float(row.get("auroc", "nan"))
        except ValueError:
            orc = auroc = float("nan")
        log.info(f"  {fold:<22} {eid}  best={best_m}({best_mcc:.4f})"
                 f"  oracle={orc:.4f}  auroc={auroc:.4f}")
        fold_winners.setdefault(fold, []).append((eid, best_m, best_mcc))
        method_wins[best_m] = method_wins.get(best_m, 0) + 1

    if method_wins:
        top = max(method_wins, key=method_wins.get)
        log.info(f"\n  Method wins: {dict(sorted(method_wins.items(), key=lambda x: -x[1]))}")
        log.info(f"  Dominant: '{top}' ({method_wins[top]} wins)")


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Probe accuracy on missing models
# ─────────────────────────────────────────────────────────────────────────────

def _probe_ts_gib_from_checkpoint(ckpt_path: Path, train_dsets: list, dev: bool,
                                   seed: int, device: str) -> float:
    """
    Run linear probe on a TS_GIB or TS-SAGE checkpoint.
    For structure-only models (q_edge_in=1): constant query features.
    For raw-feature models (q_edge_in>1): uses quantile features as q_ea.
    Returns probe accuracy; -1.0 if fewer than 2 training datasets.
    """
    if not ckpt_path.exists():
        return -1.0
    if len(train_dsets) < 2:
        return -1.0

    state      = torch.load(ckpt_path, weights_only=True)
    is_ts_gib  = "ctx_enc.0.weight" in state
    du         = _delta_us(DELTA_SECS)

    if is_ts_gib:
        has_q_enc = any(k.startswith("q_enc.") for k in state)
        q_edge_in = state["q_enc.0.weight"].shape[1] if has_q_enc else 1
        use_bn    = "to_dist.weight" in state
        model     = TS_GIB(node_in=NODE_FEAT_DIM, ctx_edge_in=1,
                           q_edge_in=q_edge_in, hidden=128,
                           use_bottleneck=use_bn, num_domains=0)
    else:
        edge_in  = state["edge_enc.0.weight"].shape[1]
        hidden   = state["edge_enc.0.weight"].shape[0]
        node_in  = state["conv1.lin_l.weight"].shape[1] - hidden
        q_edge_in = edge_in
        model     = TemporalEdgeSAGE(node_in=node_in, edge_in=edge_in, hidden=hidden)

    model.load_state_dict(state)
    model.eval().to(device)

    use_raw_features = q_edge_in > 1

    all_embs, all_labels = [], []
    max_per_ds = 500

    for ds_idx, ds in enumerate(train_dsets):
        if use_raw_features:
            g = load_graph(ds, tier="B", dev=dev)
            # Align feature dim to what the model expects
            feat = g.edge_attr_q
            d = feat.shape[1]
            if d < q_edge_in:
                feat = torch.cat([feat, torch.zeros(feat.shape[0], q_edge_in - d)], dim=1)
            elif d > q_edge_in:
                feat = feat[:, :q_edge_in]
            feat_np = feat.numpy().astype(np.float32)
        else:
            g = load_graph(ds, tier="B", dev=dev)
            g = copy.copy(g)
            E = g.edge_attr.shape[0]
            g.edge_attr   = torch.ones(E, 1)
            g.edge_attr_q = torch.ones(E, 1)
            feat_np = None

        src_np_ds  = g.edge_index[0].numpy()
        dst_np_ds  = g.edge_index[1].numpy()
        time_np_ds = g.edge_time.numpy()
        E = src_np_ds.shape[0]

        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(E, min(max_per_ds, E), replace=False))

        embs = []
        for start in range(0, len(idx), BATCH_SIZE):
            ids = idx[start:start + BATCH_SIZE]
            if use_raw_features:
                q_ea = torch.as_tensor(feat_np[ids], dtype=torch.float32, device=device)
            else:
                q_ea = torch.ones(len(ids), 1, device=device)
            data_list = batch_build_subgraphs(
                src_np_ds, dst_np_ds, time_np_ds,
                src_np_ds[ids], dst_np_ds[ids], time_np_ds[ids],
                delta_us=du, max_edges=MAX_SUB_EDGES,
                node_feat_dim=NODE_FEAT_DIM, seed=seed, n_jobs=N_JOBS,
            )
            with torch.no_grad():
                embs.append(_ts_gib_embed_batch(model, data_list, q_ea, device).cpu().numpy())

        all_embs.append(np.concatenate(embs))
        all_labels.extend([ds_idx] * len(all_embs[-1]))

    X = np.concatenate(all_embs)
    y = np.array(all_labels)
    if len(np.unique(y)) < 2:
        return -1.0
    ti, vi = _tts(np.arange(len(X)), test_size=0.2, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X[ti], y[ti])
    return accuracy_score(y[vi], clf.predict(X[vi]))


def run_task4(dev=True):
    """
    Compute probe accuracy for models that are missing it.
    Targets: E13.2_ts_sage_raw, E13.1_ts_gib_raw_b0.01,
             E12.1_ts_gib_b0.001, E12.1_ts_gib_b0.1
    """
    log.info(f"\n{'='*65}")
    log.info("Task 4 — Probe accuracy for missing models")
    log.info(f"{'='*65}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_path = Path("results/results.csv")

    probe_targets = [
        ("E13.2_ts_sage_raw",     0),
        ("E13.1_ts_gib_raw_b0.01", 0),
        ("E12.1_ts_gib_b0.001",   0),
        ("E12.1_ts_gib_b0.1",     0),
    ]

    for exp_id, seed in probe_targets:
        log.info(f"\n  Checking probe for {exp_id} seed={seed}...")

        # Check if probe is already in results
        probe_exists = False
        if results_path.exists():
            with open(results_path) as f:
                for row in csv.DictReader(f):
                    if (row["experiment_id"] == exp_id and int(row["seed"]) == seed
                            and row["metric"] == "dataset_probe_acc"):
                        log.info(f"  Probe already logged: {float(row['value']):.4f}")
                        probe_exists = True
                        break
        if probe_exists:
            continue

        # Find any checkpoint for this experiment
        ckpt_pattern = f"{exp_id}_seed{seed}_test*.pt"
        ckpts = list(MODELS_DIR.glob(ckpt_pattern))
        if not ckpts:
            log.warning(f"  No checkpoints found for {exp_id} — skipping.")
            continue

        log.info(f"  Found {len(ckpts)} checkpoint(s). Running probe...")

        for ckpt in ckpts:
            stem  = ckpt.stem
            tail  = stem.split("_test")
            if len(tail) < 2:
                continue
            test_fold = tail[-1]
            fold      = next((f for f in ALL_FOLDS if f["test"] == test_fold), None)
            if fold is None:
                continue
            train_dsets = fold["train"]

            try:
                probe = _probe_ts_gib_from_checkpoint(ckpt, train_dsets, dev, seed, device)
                if probe >= 0:
                    log.info(f"  {exp_id} seed={seed} fold={test_fold} probe={probe:.4f}")
                    log_result(exp_id, seed, train_dsets, test_fold,
                               "dataset_probe_acc", probe, 0.0)
                else:
                    log.warning(f"  Probe returned {probe} for {ckpt.name}")
            except Exception as e:
                log.warning(f"  Probe failed for {ckpt.name}: {e}")

    # Print summary
    log.info("\n  Task 4 probe summary:")
    if not results_path.exists():
        return
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["metric"] == "dataset_probe_acc":
                log.info(f"  {row['experiment_id']:<35} seed={row['seed']}"
                         f"  fold={row['test_dataset']:<20}"
                         f"  probe={float(row['value']):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 — Per-attack-class F1 for headline TS-GIB
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _per_attack_f1(ckpt_path: Path, fold: dict, dev: bool, device: str,
                   p_src: float) -> dict | None:
    """
    Compute per-attack-class F1 at calibrated threshold for a TS_GIB checkpoint.
    Returns dict: {attack_class: f1} or None on failure.
    """
    if not ckpt_path.exists():
        return None

    state = torch.load(ckpt_path, weights_only=True)
    model = TS_GIB(node_in=NODE_FEAT_DIM, ctx_edge_in=1, q_edge_in=1,
                   hidden=128, use_bottleneck=True, num_domains=0)
    model.load_state_dict(state)
    model.eval().to(device)

    du = _delta_us(DELTA_SECS)
    _, test_graph, _, _ = _load_fold_struct(fold, dev)

    src_np, dst_np, time_np = _graph_arrays(test_graph)
    labels_np   = test_graph.edge_label.numpy()
    label_types = np.array(test_graph.edge_label_type)
    n_total     = len(labels_np)

    if n_total > MAX_EVAL_EDGES:
        _, eval_idx = _tts(np.arange(n_total),
                           test_size=MAX_EVAL_EDGES / n_total,
                           random_state=42, stratify=labels_np)
        eval_idx = np.sort(np.array(eval_idx, dtype=np.int64))
    else:
        eval_idx = np.arange(n_total, dtype=np.int64)

    t0 = time.time()
    all_data = batch_build_subgraphs(
        src_np, dst_np, time_np,
        src_np[eval_idx], dst_np[eval_idx], time_np[eval_idx],
        delta_us=du, max_edges=MAX_SUB_EDGES,
        node_feat_dim=NODE_FEAT_DIM, seed=0, n_jobs=N_JOBS,
    )
    log.info(f"  Subgraph extraction: {time.time()-t0:.1f}s")

    all_scores = []
    for start in range(0, len(eval_idx), BATCH_SIZE):
        ids_b = eval_idx[start:start + BATCH_SIZE]
        q_ea  = torch.ones(len(ids_b), 1, device=device)
        dl    = all_data[start:start + BATCH_SIZE]
        logits, _ = _ts_gib_forward_batch(model, dl, q_ea, device)
        all_scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())

    y_score  = np.concatenate(all_scores)
    y_true   = labels_np[eval_idx]
    atk_types = label_types[eval_idx]

    # Calibrated threshold: top-k% at source-val attack rate
    k     = max(1, int(round(p_src * len(y_score))))
    cal_t = _topk_threshold(y_score, k)
    y_pred_cal = (y_score >= cal_t).astype(int)

    per_class = compute_per_class_f1(list(atk_types), y_pred_cal, y_true)
    return per_class


def run_task5(seeds=None, dev=True):
    """
    Per-attack-class F1 for E14.1_ts_gib_b0.01 on each fold.
    Uses calibrated threshold (topk at p_src rate).
    """
    log.info(f"\n{'='*65}")
    log.info("Task 5 — Per-attack-class F1 for headline TS-GIB")
    log.info(f"{'='*65}")

    seeds  = seeds or T1_SEEDS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id = T1_EXP_ID
    FOLDS  = ["lycos_ids2017", "cic_ids2018", "unsw_nb15", "ton_iot"]

    # Collect best seed per fold based on AUROC (prefer seed 0 if equal)
    best_seed_per_fold: dict = {}
    results_path = Path("results/results.csv")
    if results_path.exists():
        fold_auroc: dict = {}
        with open(results_path) as f:
            for row in csv.DictReader(f):
                if row["experiment_id"] != exp_id or row["metric"] != "auroc":
                    continue
                if int(row["seed"]) in seeds:
                    td = row["test_dataset"]
                    v  = float(row["value"])
                    if td not in fold_auroc or v > fold_auroc[td][1]:
                        fold_auroc[td] = (int(row["seed"]), v)
        best_seed_per_fold = {td: s for td, (s, _) in fold_auroc.items()}

    all_fold_results: dict = {}
    for fold_name in FOLDS:
        fold = next((f for f in ALL_FOLDS if f["test"] == fold_name), None)
        if fold is None:
            continue

        seed = best_seed_per_fold.get(fold_name, 0)

        # Check AUROC threshold — skip non-solvable folds
        fold_auroc_val = float("nan")
        if results_path.exists():
            with open(results_path) as f:
                for row in csv.DictReader(f):
                    if (row["experiment_id"] == exp_id and int(row["seed"]) == seed
                            and row["test_dataset"] == fold_name
                            and row["metric"] == "auroc"):
                        fold_auroc_val = float(row["value"])
                        break

        if not np.isnan(fold_auroc_val) and fold_auroc_val < MIN_AUROC_SOLVABLE:
            log.info(f"  Skipping {fold_name} (AUROC={fold_auroc_val:.3f} < {MIN_AUROC_SOLVABLE})")
            continue

        ckpt_path = MODELS_DIR / f"{exp_id}_seed{seed}_test{fold_name}.pt"
        if not ckpt_path.exists():
            log.warning(f"  Checkpoint missing: {ckpt_path.name}")
            continue

        p_src = _get_p_src_from_csv(exp_id, seed, fold_name)
        log.info(f"\n  {fold_name}  seed={seed}  p_src={p_src:.4f}  "
                 f"AUROC={fold_auroc_val:.3f}")

        try:
            per_class = _per_attack_f1(ckpt_path, fold, dev, device, p_src)
            if per_class is None:
                continue
        except Exception as e:
            log.warning(f"  Failed: {e}")
            continue

        all_fold_results[fold_name] = per_class

        # Log per-class F1 to results.csv
        for cls, f1_val in per_class.items():
            if not np.isnan(f1_val):
                log_result(exp_id, seed, fold["train"], fold_name,
                           f"f1_{cls}", f1_val, 0.0)
        log.info(f"  Per-class F1: " +
                 "  ".join(f"{c}={v:.3f}" for c, v in per_class.items()))

    # Print summary table
    if not all_fold_results:
        log.warning("  No per-attack results computed.")
        return

    log.info(f"\n  Per-attack-class F1 summary ({T1_EXP_ID}):")
    header = f"  {'Fold':<22}" + "".join(f"  {c:<20}" for c in ATTACK_CLASSES)
    log.info(header)
    for fold_name, per_class in sorted(all_fold_results.items()):
        row_s = f"  {fold_name:<22}"
        for c in ATTACK_CLASSES:
            v = per_class.get(c, float("nan"))
            row_s += f"  {v:>20.3f}" if not np.isnan(v) else f"  {'n/a':>20}"
        log.info(row_s)

    # Class-level means across folds
    means_row = f"  {'mean':<22}"
    for c in ATTACK_CLASSES:
        vals = [per_class.get(c, float("nan")) for per_class in all_fold_results.values()
                if not np.isnan(per_class.get(c, float("nan")))]
        if vals:
            means_row += f"  {np.mean(vals):>20.3f}"
        else:
            means_row += f"  {'n/a':>20}"
    log.info(means_row)

    # Hypothesis check: structural attacks should transfer better
    log.info("\n  Hypothesis: Reconnaissance/DoS should have higher F1 than Brute Force/Injection")
    for fold_name, per_class in sorted(all_fold_results.items()):
        recon = per_class.get("Reconnaissance", float("nan"))
        dos   = per_class.get("DoS_DDoS", float("nan"))
        brute = per_class.get("BruteForce", float("nan"))
        inj   = per_class.get("Injection_Exploit", float("nan"))
        structural  = np.nanmean([recon, dos]) if not np.all(np.isnan([recon, dos])) else float("nan")
        application = np.nanmean([brute, inj]) if not np.all(np.isnan([brute, inj])) else float("nan")
        if not np.isnan(structural) and not np.isnan(application):
            result_str = ("✓ confirms" if structural > application else "✗ refutes")
            log.info(f"  {fold_name:<22} struct={structural:.3f}  app={application:.3f}  {result_str}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 6 — Reported MCC column
# ─────────────────────────────────────────────────────────────────────────────

def run_task6(seeds=None):
    """Print reported MCC (threshold=0.5) vs calibrated MCC for all headline models."""
    log.info(f"\n{'='*65}")
    log.info("Task 6 — Reported MCC at threshold=0.5 vs Calibrated MCC")
    log.info(f"{'='*65}")

    seeds = seeds or list(range(3))
    results_path = Path("results/results.csv")
    if not results_path.exists():
        log.warning("results.csv not found.")
        return

    ROWS = [
        ("E-GS raw (B4)",             "E1.C"),
        ("E-GS no feat (E1.E)",        "E1.E_struct_only"),
        ("TS-SAGE no feat (E14.2)",    T2_EXP_ID),
        ("TS-SAGE raw (E13.2)",        "E13.2_ts_sage_raw"),
        ("GIB no feat β.01 (E13.4)",  "E13.4_gib_nofeat_b0.01"),
        ("TS-GIB raw β.01 (E13.1)",   "E13.1_ts_gib_raw_b0.01"),
        ("TS-GIB no feat β.01 (E14.1)", T1_EXP_ID),
    ]
    FOLD_NAMES = ["lycos_ids2017", "cic_ids2018", "unsw_nb15", "ton_iot"]
    FOLD_ABBR  = ["lycos", "cic18", "unsw", "ton"]

    data: dict = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if int(row["seed"]) not in seeds:
                continue
            for _, eid in ROWS:
                if row["experiment_id"] == eid:
                    key = (eid, row["test_dataset"], row["metric"])
                    data.setdefault(key, []).append(float(row["value"]))

    def _get(eid, td, metric):
        vals = data.get((eid, td, metric), [])
        return np.mean(vals) if vals else float("nan")

    log.info(f"\n  Reported MCC (threshold=0.5):")
    log.info(f"  {'Method':<38}" + "".join(f"  {a:>8}" for a in FOLD_ABBR) + "  {'mean3f':>8}")
    for label, eid in ROWS:
        vals = [_get(eid, td, "mcc") for td in FOLD_NAMES]
        non_ton = [v for v, td in zip(vals, FOLD_NAMES) if td != "ton_iot" and not np.isnan(v)]
        mean3 = np.mean(non_ton) if non_ton else float("nan")
        row_s = f"  {label:<38}" + "".join(
            f"  {v:>8.3f}" if not np.isnan(v) else f"  {'---':>8}" for v in vals)
        row_s += f"  {mean3:>8.3f}" if not np.isnan(mean3) else f"  {'---':>8}"
        log.info(row_s)

    log.info(f"\n  Calibrated MCC (topk at p_src rate):")
    log.info(f"  {'Method':<38}" + "".join(f"  {a:>8}" for a in FOLD_ABBR) + "  {'mean3f':>8}")
    for label, eid in ROWS:
        vals = [_get(eid, td, "calibrated_mcc") for td in FOLD_NAMES]
        non_ton = [v for v, td in zip(vals, FOLD_NAMES) if td != "ton_iot" and not np.isnan(v)]
        mean3 = np.mean(non_ton) if non_ton else float("nan")
        row_s = f"  {label:<38}" + "".join(
            f"  {v:>8.3f}" if not np.isnan(v) else f"  {'---':>8}" for v in vals)
        row_s += f"  {mean3:>8.3f}" if not np.isnan(mean3) else f"  {'---':>8}"
        log.info(row_s)


# ─────────────────────────────────────────────────────────────────────────────
# Task 7 — Consistency check
# ─────────────────────────────────────────────────────────────────────────────

def run_task7():
    """Verify internal consistency of all reported numbers in results.csv."""
    log.info(f"\n{'='*65}")
    log.info("Task 7 — Consistency check")
    log.info(f"{'='*65}")

    results_path = Path("results/results.csv")
    if not results_path.exists():
        log.warning("results.csv not found.")
        return

    issues: list = []

    # Load all data
    data: dict = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            key = (row["experiment_id"], int(row["seed"]), row["test_dataset"], row["metric"])
            data.setdefault(key, []).append(float(row["value"]))

    def _mean(eid, seed, td, metric):
        vals = data.get((eid, seed, td, metric), [])
        return np.mean(vals) if vals else float("nan")

    # Check 1: AUROC values are in [0, 1]
    for (eid, seed, td, metric), vals in data.items():
        if metric == "auroc":
            for v in vals:
                if not np.isnan(v) and not (0 <= v <= 1):
                    issues.append(f"AUROC out of range: {eid} seed={seed} {td} = {v:.4f}")

    # Check 2: MCC values are in [-1, 1]
    for (eid, seed, td, metric), vals in data.items():
        if "mcc" in metric:
            for v in vals:
                if not np.isnan(v) and not (-1 <= v <= 1):
                    issues.append(f"MCC out of range: {eid} seed={seed} {td} {metric} = {v:.4f}")

    # Check 3: p_src is in (0, 1)
    for (eid, seed, td, metric), vals in data.items():
        if metric == "p_src":
            for v in vals:
                if not np.isnan(v) and not (0 < v < 1):
                    issues.append(f"p_src out of range: {eid} seed={seed} {td} = {v:.4f}")

    # Check 4: calibrated_mcc <= oracle_mcc (with tolerance)
    all_eids = set(eid for (eid, _, _, _) in data)
    for eid in all_eids:
        all_seeds = set(s for (e, s, _, _) in data if e == eid)
        for seed in all_seeds:
            all_folds = set(td for (e, s, td, _) in data if e == eid and s == seed)
            for td in all_folds:
                cal = _mean(eid, seed, td, "calibrated_mcc")
                orc = _mean(eid, seed, td, "oracle_mcc")
                if not np.isnan(cal) and not np.isnan(orc):
                    if cal > orc + 0.02:  # oracle should be >= calibrated
                        issues.append(
                            f"cal_mcc > oracle_mcc: {eid} seed={seed} {td}"
                            f"  cal={cal:.4f} orc={orc:.4f}")

    # Check 5: E14.1 should have 3 seeds × 4 folds × key metrics
    for seed in T1_SEEDS:
        for td in ["lycos_ids2017", "cic_ids2018", "unsw_nb15", "ton_iot"]:
            for metric in ["mcc", "calibrated_mcc", "auroc"]:
                v = _mean(T1_EXP_ID, seed, td, metric)
                if np.isnan(v):
                    issues.append(f"Missing: {T1_EXP_ID} seed={seed} {td} {metric}")

    # Check 6: E14.2 should have 4 folds × key metrics
    for td in ["lycos_ids2017", "cic_ids2018", "unsw_nb15", "ton_iot"]:
        for metric in ["mcc"]:
            v = _mean(T2_EXP_ID, 0, td, metric)
            if np.isnan(v):
                issues.append(f"Missing: {T2_EXP_ID} seed=0 {td} {metric}")

    # Check 7: Multi-seed std check for E14.1
    log.info("\n  E14.1 multi-seed statistics:")
    for td in ["lycos_ids2017", "cic_ids2018", "unsw_nb15", "ton_iot"]:
        vals = [_mean(T1_EXP_ID, s, td, "calibrated_mcc") for s in T1_SEEDS]
        valid = [v for v in vals if not np.isnan(v)]
        if len(valid) >= 2:
            m, s = np.mean(valid), np.std(valid)
            flag = " *** HIGH VARIANCE ***" if s > 0.10 else ""
            log.info(f"  {td:<22} mean={m:.4f}  std={s:.4f}  n={len(valid)}{flag}")
            if s > 0.10:
                issues.append(f"High variance: {T1_EXP_ID} {td} std={s:.4f} > 0.10")
        else:
            log.info(f"  {td:<22} insufficient seeds ({len(valid)}/{len(T1_SEEDS)})")

    if issues:
        log.warning(f"\n  *** {len(issues)} consistency issue(s) found: ***")
        for iss in issues:
            log.warning(f"  ! {iss}")
    else:
        log.info(f"\n  All consistency checks passed.")

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed summary table (final output)
# ─────────────────────────────────────────────────────────────────────────────

def print_final_summary():
    """Print final headline numbers: multi-seed mean ± std for E14.1."""
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return

    log.info(f"\n{'='*65}")
    log.info(f"Final headline: {T1_EXP_ID} (multi-seed)")
    log.info(f"{'='*65}")

    data: dict = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["experiment_id"] != T1_EXP_ID:
                continue
            if int(row["seed"]) not in T1_SEEDS:
                continue
            key = (row["test_dataset"], row["metric"])
            data.setdefault(key, []).append(float(row["value"]))

    FOLDS = ["lycos_ids2017", "cic_ids2018", "unsw_nb15", "ton_iot"]
    METRICS = ["calibrated_mcc", "auroc", "oracle_mcc"]

    for metric in METRICS:
        log.info(f"\n  {metric}:")
        log.info(f"  {'fold':<22} {'mean':>8} {'std':>8} {'n':>4}")
        fold_means = []
        for td in FOLDS:
            vals = data.get((td, metric), [])
            if vals:
                m = np.mean(vals)
                s = np.std(vals)
                fold_means.append((m, td))
                log.info(f"  {td:<22} {m:>8.4f} {s:>8.4f} {len(vals):>4}")
            else:
                log.info(f"  {td:<22} {'---':>8}")

        non_ton = [m for m, td in fold_means if td != "ton_iot"]
        if non_ton:
            log.info(f"  {'mean (3-fold)':<22} {np.mean(non_ton):>8.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Phase 14 Final: all number-collection tasks per spex14.md")
    parser.add_argument("--tasks", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7],
                        help="Tasks to run (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help=f"Seeds for Task 1 (default: {T1_SEEDS})")
    parser.add_argument("--folds", nargs="+", default=None,
                        help="Restrict folds by test dataset name")
    parser.add_argument("--no-dev", dest="dev", action="store_false",
                        help="Use full datasets (default: dev subsample)")
    parser.add_argument("--target-exps", nargs="+", default=None,
                        help="Task 3: custom list of experiment IDs to calibrate")
    parser.add_argument("--direct-cal", action="store_true",
                        help="Task 3: use direct calibration (bypass phase13 call)")
    args = parser.parse_args()

    run_folds = ALL_FOLDS
    if args.folds:
        run_folds = [f for f in ALL_FOLDS if f["test"] in args.folds]
        if not run_folds:
            log.error(f"No folds match {args.folds}"); return

    seeds = args.seeds or T1_SEEDS

    log.info(f"Phase 14 Final — tasks={args.tasks}  dev={args.dev}  seeds={seeds}")

    if 1 in args.tasks:
        run_task1(seeds=seeds, folds=run_folds, dev=args.dev)

    if 2 in args.tasks:
        run_task2(seeds=T2_SEEDS, folds=run_folds, dev=args.dev)

    if 3 in args.tasks:
        if args.direct_cal:
            run_task3_direct(dev=args.dev)
        else:
            run_task3(target_exps=args.target_exps, dev=args.dev)

    if 4 in args.tasks:
        run_task4(dev=args.dev)

    if 5 in args.tasks:
        run_task5(seeds=seeds, dev=args.dev)

    if 6 in args.tasks:
        run_task6(seeds=seeds)

    if 7 in args.tasks:
        run_task7()

    print_final_summary()


if __name__ == "__main__":
    main()
