#!/usr/bin/env python3
"""
Phase 9 (spex9.md): Consolidate TS-SAGE on realistic pairwise transfer.

E9.1 — Pairwise CIC17↔CIC18: TS-SAGE vs E1.E, 3 seeds each direction
E9.2 — Δ sweep on best pairwise direction, single seed
E9.3 — Per-attack F1 on best E9.1 TS-SAGE checkpoint (inference only)
E9.4 — {CIC17, CIC18}→UNSW and →ToN with TS-SAGE, 3 seeds
E9.5 — E1.E baseline for E9.4: {CIC17, CIC18}→UNSW

Usage:
    python scripts/run_phase9.py --exp e9_1 [--seeds 0 1 2] [--delta 60]
    python scripts/run_phase9.py --exp e9_2 [--seed 0]
    python scripts/run_phase9.py --exp e9_3 [--seed 0] [--delta 60]
    python scripts/run_phase9.py --exp e9_4 [--seeds 0 1 2] [--delta 60]
    python scripts/run_phase9.py --exp e9_5 [--seeds 0 1 2]
    python scripts/run_phase9.py --exp all  [--seeds 0 1 2]

    Add --no-dev to use full datasets (default: dev subsamples).
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
from sklearn.model_selection import train_test_split as _tts

from run_phase8 import (
    _load_fold_struct, _train_temporal, _eval_temporal,
    _probe_on_temporal_encoder,
    NODE_FEAT_DIM, BATCH_SIZE, MAX_TRAIN_EDGES,
    MAX_EPOCHS, PATIENCE, _delta_us,
)
from run_phase4 import _make_struct_only

from src.utils.logging import setup_logging, log_result, already_done, save_model, MODELS_DIR
from src.utils.seeding import seed_everything
from src.utils.metrics import compute_all_metrics, compute_mcc
from src.data.graph_builder import load_graph, combine_graphs
from src.models.egraphsage import EdgeAwareSAGE
from src.models.temporal_gnn import TemporalEdgeSAGE
from src.train.train_loops import train_egraphsage
from src.train.eval import eval_egraphsage

log = logging.getLogger(__name__)

FIGURES_DIR    = Path("results/figures/phase9")
DEFAULT_DELTA  = 60
DELTA_SWEEP_E9 = [15, 60, 300, 1800]

# CIC17 = lycos_ids2017, CIC18 = cic_ids2018
PAIRWISE_FOLDS = [
    {"train": ["cic_ids2018"],   "test": "lycos_ids2017"},   # CIC18 → CIC17
    {"train": ["lycos_ids2017"], "test": "cic_ids2018"},     # CIC17 → CIC18
]

TRIPLE_SOURCE_FOLDS = [
    {"train": ["lycos_ids2017", "cic_ids2018"], "test": "unsw_nb15"},
    {"train": ["lycos_ids2017", "cic_ids2018"], "test": "ton_iot"},
]


# ── Summary helpers ────────────────────────────────────────────────────────────

def _print_summary(exp_id: str, seeds, label: str = ""):
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
    if not fold_vals:
        return
    tag = f"  {exp_id}" + (f" ({label})" if label else "") + " summary:"
    log.info(tag)
    fold_means = []
    for td, vals in sorted(fold_vals.items()):
        m, s = np.mean(vals), np.std(vals)
        fold_means.append(m)
        log.info(f"    {td:<22} mean={m:.4f}  std={s:.4f}  n={len(vals)}")
    log.info(f"  Overall mean MCC: {np.mean(fold_means):.4f}")


def _print_delta_sweep_summary(seed: int):
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return
    log.info(f"\n  E9.2 Δ sweep summary (seed={seed}):")
    delta_vals: dict = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if not row["experiment_id"].startswith("E9.2_ts_sage_d"):
                continue
            if row["metric"] != "mcc" or int(row["seed"]) != seed:
                continue
            delta = int(row["experiment_id"].split("_d")[-1])
            delta_vals.setdefault(delta, []).append(float(row["value"]))
    for d in sorted(delta_vals):
        vals = delta_vals[d]
        log.info(f"    Δ={d:5d}s  mean_mcc={np.mean(vals):.4f}  folds={len(vals)}")


def _get_best_pairwise_fold(seeds):
    """Return the PAIRWISE_FOLD with highest mean MCC from E9.1 TS-SAGE results."""
    results_path = Path("results/results.csv")
    exp_id = f"E9.1_ts_sage_d{DEFAULT_DELTA}"
    fold_means: dict = {}
    if results_path.exists():
        with open(results_path) as f:
            for row in csv.DictReader(f):
                if row["experiment_id"] != exp_id or row["metric"] != "mcc":
                    continue
                if int(row["seed"]) not in seeds:
                    continue
                fold_means.setdefault(row["test_dataset"], []).append(float(row["value"]))

    if not fold_means:
        log.warning("  No E9.1 TS-SAGE results; defaulting to CIC17→CIC18 fold")
        return PAIRWISE_FOLDS[1]

    best_test = max(fold_means, key=lambda k: np.mean(fold_means[k]))
    best_mean = np.mean(fold_means[best_test])
    log.info(f"  Best E9.1 fold: test={best_test}  mean_mcc={best_mean:.4f}")
    return next(f for f in PAIRWISE_FOLDS if f["test"] == best_test)


# ── Core fold runners ──────────────────────────────────────────────────────────

def _run_ts_sage_fold(exp_id: str, fold: dict, seed: int,
                      delta_secs: int, dev: bool) -> "TemporalEdgeSAGE | None":
    """Train and evaluate TS-SAGE on a single fold. Returns model if trained, else None."""
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    du          = _delta_us(delta_secs)
    train_dsets = fold["train"]
    test_dset   = fold["test"]

    if already_done(exp_id, seed, test_dset):
        log.info(f"  Skipping {exp_id} seed={seed} test={test_dset}")
        return None

    seed_everything(seed)
    t0 = time.time()
    log.info(f"\n  [{exp_id}] train={train_dsets}  test={test_dset}  seed={seed}")

    combined, test_graph, _, _ = _load_fold_struct(fold, dev)

    model = TemporalEdgeSAGE(
        node_in=NODE_FEAT_DIM, edge_in=1, hidden=128, num_classes=2
    )
    model, _ = _train_temporal(
        model, combined, device, seed, exp_id, test_dset,
        delta_us=du, epochs=MAX_EPOCHS, patience=PATIENCE,
        batch_size=BATCH_SIZE, max_train_edges=MAX_TRAIN_EDGES,
    )

    result  = _eval_temporal(model, test_graph, device, delta_us=du)
    metrics = compute_all_metrics(result["y_true"], result["y_pred"],
                                  y_true_type=test_graph.edge_label_type)
    elapsed = time.time() - t0

    log.info(f"  {exp_id} seed={seed} test={test_dset}"
             f"  MCC={metrics['mcc']:.4f}  macro_F1={metrics['macro_f1']:.4f}")
    log_result(exp_id, seed, train_dsets, test_dset, "mcc",        metrics["mcc"],        elapsed)
    log_result(exp_id, seed, train_dsets, test_dset, "macro_f1",   metrics["macro_f1"],   elapsed)
    log_result(exp_id, seed, train_dsets, test_dset, "delta_secs", float(delta_secs),     0.0)
    for cls, f1 in metrics.get("per_class_f1", {}).items():
        log_result(exp_id, seed, train_dsets, test_dset, f"f1_{cls}", f1, elapsed)

    return model


def _run_e1e_fold(exp_id: str, fold: dict, seed: int, dev: bool):
    """Train and evaluate structure-only E-GraphSAGE (E1.E style) on a single fold."""
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    train_dsets = fold["train"]
    test_dset   = fold["test"]

    if already_done(exp_id, seed, test_dset):
        log.info(f"  Skipping {exp_id} seed={seed} test={test_dset}")
        return

    seed_everything(seed)
    t0 = time.time()
    log.info(f"\n  [{exp_id}] train={train_dsets}  test={test_dset}  seed={seed}")

    train_graphs = [_make_struct_only(load_graph(ds, tier="B", dev=dev))
                    for ds in train_dsets]
    combined   = combine_graphs(train_graphs)
    test_graph = _make_struct_only(load_graph(test_dset, tier="B", dev=dev))

    n = combined.edge_label.shape[0]
    ti, vi = _tts(np.arange(n), test_size=0.2, random_state=seed,
                  stratify=combined.edge_label.numpy())
    val_split = {"train": ti.tolist(), "val": vi.tolist()}

    model = EdgeAwareSAGE(
        node_in=combined.x.shape[1],
        edge_in=combined.edge_attr.shape[1],
    )
    best_state = train_egraphsage(
        model, combined, val_split=val_split, device=device, use_quantile=False,
    )
    model.load_state_dict(best_state)
    save_model(exp_id, seed, test_dset, best_state)

    result  = eval_egraphsage(model, test_graph, device=device, use_quantile=False)
    metrics = compute_all_metrics(result["y_true"], result["y_pred"],
                                  y_true_type=test_graph.edge_label_type)
    elapsed = time.time() - t0

    log.info(f"  {exp_id} seed={seed} test={test_dset}"
             f"  MCC={metrics['mcc']:.4f}  macro_F1={metrics['macro_f1']:.4f}")
    log_result(exp_id, seed, train_dsets, test_dset, "mcc",      metrics["mcc"],      elapsed)
    log_result(exp_id, seed, train_dsets, test_dset, "macro_f1", metrics["macro_f1"], elapsed)
    for cls, f1 in metrics.get("per_class_f1", {}).items():
        log_result(exp_id, seed, train_dsets, test_dset, f"f1_{cls}", f1, elapsed)


# ── E9.1 — Pairwise CIC17↔CIC18 ──────────────────────────────────────────────

def run_e9_1(seeds, delta_secs: int, dev: bool):
    """E9.1: Pairwise CIC17↔CIC18, TS-SAGE vs E1.E, 3 seeds."""
    log.info(f"=== E9.1  Pairwise CIC17↔CIC18  Δ={delta_secs}s  seeds={seeds} ===")
    exp_ts  = f"E9.1_ts_sage_d{delta_secs}"
    exp_e1e = "E9.1_e1e"
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    du      = _delta_us(delta_secs)

    for seed in seeds:
        for fold in PAIRWISE_FOLDS:
            # TS-SAGE
            model = _run_ts_sage_fold(exp_ts, fold, seed, delta_secs, dev)

            # Probe diagnostic — only meaningful with ≥2 training datasets
            if model is not None and len(fold["train"]) >= 2:
                train_dsets = fold["train"]
                test_dset   = fold["test"]
                probe_acc   = _probe_on_temporal_encoder(
                    model, train_dsets, dev, seed, device, du)
                log.info(f"  Probe accuracy: {probe_acc:.4f}")
                log_result(exp_ts, seed, train_dsets, test_dset,
                           "dataset_probe_acc", probe_acc, 0.0)
            elif model is not None:
                log.info("  Probe skipped (single training source).")

            # E1.E baseline
            _run_e1e_fold(exp_e1e, fold, seed, dev)

    _print_summary(exp_ts,  seeds, "TS-SAGE")
    _print_summary(exp_e1e, seeds, "E1.E")


# ── E9.2 — Δ sweep on best pairwise direction ─────────────────────────────────

def run_e9_2(seed: int, dev: bool):
    """E9.2: Δ sweep on the best E9.1 direction, single seed."""
    log.info(f"=== E9.2  Δ sweep  deltas={DELTA_SWEEP_E9}  seed={seed} ===")
    best_fold = _get_best_pairwise_fold([seed])
    log.info(f"  Sweeping on fold: test={best_fold['test']}")

    for delta in DELTA_SWEEP_E9:
        exp_id = f"E9.2_ts_sage_d{delta}"
        log.info(f"\n--- Δ = {delta}s ---")
        _run_ts_sage_fold(exp_id, best_fold, seed, delta, dev)

    _print_delta_sweep_summary(seed)


# ── E9.3 — Per-attack F1 on best E9.1 ────────────────────────────────────────

def run_e9_3(seed: int, dev: bool, delta_secs: int = DEFAULT_DELTA):
    """E9.3: Per-attack F1 for each E9.1 TS-SAGE checkpoint (inference only)."""
    log.info("=== E9.3  Per-attack F1 on E9.1 TS-SAGE checkpoints ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    du     = _delta_us(delta_secs)
    exp_id = f"E9.1_ts_sage_d{delta_secs}"

    for fold in PAIRWISE_FOLDS:
        test_dset   = fold["test"]
        train_dsets = fold["train"]
        enc_path    = MODELS_DIR / f"{exp_id}_seed{seed}_test{test_dset}.pt"

        if not enc_path.exists():
            log.warning(f"  Missing checkpoint: {enc_path}  (run e9_1 first)")
            continue

        model = TemporalEdgeSAGE(node_in=NODE_FEAT_DIM, edge_in=1, hidden=128)
        model.load_state_dict(torch.load(enc_path, weights_only=True))

        _, test_graph, _, _ = _load_fold_struct(fold, dev)
        result  = _eval_temporal(model, test_graph, device, delta_us=du)
        metrics = compute_all_metrics(result["y_true"], result["y_pred"],
                                      y_true_type=test_graph.edge_label_type)

        log.info(f"\n  fold={test_dset}  MCC={metrics['mcc']:.4f}")
        for cls, f1 in metrics.get("per_class_f1", {}).items():
            log_result("E9.3_per_attack", seed, train_dsets, test_dset, f"f1_{cls}", f1, 0.0)

        log.info("  Per-class F1:")
        for cls, f1 in sorted(metrics.get("per_class_f1", {}).items(),
                               key=lambda x: x[1], reverse=True):
            log.info(f"    {cls:<24}  {f1:.4f}")


# ── E9.4 — Triple-source: {CIC17, CIC18} → UNSW / ToN ────────────────────────

def run_e9_4(seeds, delta_secs: int, dev: bool):
    """E9.4: Train on {CIC17, CIC18}, test on UNSW and ToN with TS-SAGE."""
    log.info(f"=== E9.4  {{CIC17,CIC18}}→UNSW/ToN  Δ={delta_secs}s  seeds={seeds} ===")
    exp_id = f"E9.4_ts_sage_d{delta_secs}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    du     = _delta_us(delta_secs)

    for seed in seeds:
        for fold in TRIPLE_SOURCE_FOLDS:
            model = _run_ts_sage_fold(exp_id, fold, seed, delta_secs, dev)

            if model is not None:
                train_dsets = fold["train"]
                test_dset   = fold["test"]
                probe_acc   = _probe_on_temporal_encoder(
                    model, train_dsets, dev, seed, device, du)
                log.info(f"  Probe accuracy: {probe_acc:.4f}")
                log_result(exp_id, seed, train_dsets, test_dset,
                           "dataset_probe_acc", probe_acc, 0.0)

    _print_summary(exp_id, seeds, "TS-SAGE {CIC17+CIC18}→UNSW/ToN")


# ── E9.5 — E1.E baseline for E9.4 ────────────────────────────────────────────

def run_e9_5(seeds, dev: bool):
    """E9.5: E1.E structure-only baseline for {CIC17, CIC18}→UNSW."""
    log.info(f"=== E9.5  E1.E {{CIC17,CIC18}}→UNSW/ToN  seeds={seeds} ===")
    exp_id = "E9.5_e1e"

    for seed in seeds:
        for fold in TRIPLE_SOURCE_FOLDS:
            _run_e1e_fold(exp_id, fold, seed, dev)

    _print_summary(exp_id, seeds, "E1.E {CIC17+CIC18}→UNSW/ToN")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Phase 9 (spex9.md)")
    parser.add_argument("--exp", required=True,
                        choices=["e9_1", "e9_2", "e9_3", "e9_4", "e9_5", "all"])
    parser.add_argument("--delta",   type=int,  default=DEFAULT_DELTA,
                        help="Temporal window in seconds (default 60)")
    parser.add_argument("--seeds",   nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seed",    type=int,  default=0,
                        help="Single seed for e9_2 and e9_3")
    parser.add_argument("--dev",    action="store_true", default=True)
    parser.add_argument("--no-dev", dest="dev", action="store_false")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.exp in ("e9_1", "all"):
        run_e9_1(args.seeds, args.delta, args.dev)

    if args.exp in ("e9_2", "all"):
        run_e9_2(args.seed, args.dev)

    if args.exp in ("e9_3", "all"):
        run_e9_3(args.seed, args.dev, args.delta)

    if args.exp in ("e9_4", "all"):
        run_e9_4(args.seeds, args.delta, args.dev)

    if args.exp in ("e9_5", "all"):
        run_e9_5(args.seeds, args.dev)


if __name__ == "__main__":
    main()
