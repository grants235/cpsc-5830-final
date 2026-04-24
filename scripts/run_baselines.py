#!/usr/bin/env python3
"""
Baselines B1–B5 under LODO evaluation.
  B1: Random Forest on Tier-A shared features
  B2: MLP on quantile Tier-A / Tier-B features
  B3: E-GraphSAGE within-dataset (upper bound)
  B4: E-GraphSAGE LODO, raw Tier-A
  B5: E-GraphSAGE LODO, quantile Tier-A

Usage:
    python scripts/run_baselines.py [--dev] [--baselines B1 B2 ...]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from src.utils.logging import setup_logging, log_result, already_done, save_model
from src.utils.seeding import seed_everything
from src.utils.metrics import compute_all_metrics
from src.data.graph_builder import load_graph, combine_graphs, load_split
from src.models.baseline_mlp import RandomForestBaseline, MLP
from src.models.egraphsage import EdgeAwareSAGE
from src.train.train_loops import train_egraphsage
from src.train.eval import eval_egraphsage

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
DATASETS    = ["lycos_ids2017", "cic_ids2018", "unsw_nb15", "ton_iot"]
SEEDS       = [0, 1, 2]

FOLDS = [
    {"train": ["cic_ids2018", "unsw_nb15", "ton_iot"],     "test": "lycos_ids2017"},
    {"train": ["lycos_ids2017", "unsw_nb15", "ton_iot"],   "test": "cic_ids2018"},
    {"train": ["lycos_ids2017", "cic_ids2018", "ton_iot"], "test": "unsw_nb15"},
    {"train": ["lycos_ids2017", "cic_ids2018", "unsw_nb15"], "test": "ton_iot"},
]


def run_b1(dev: bool = True):
    """B1 — Random Forest on Tier-A shared features."""
    for fold_idx, fold in enumerate(FOLDS):
        train_dsets = fold["train"]
        test_dset   = fold["test"]
        for seed in SEEDS:
            if already_done("B1_RF", seed, test_dset):
                log.info(f"Skipping B1_RF seed={seed} test={test_dset} (already done)")
                continue
            seed_everything(seed)
            t0 = time.time()

            # Collect training Tier-A features
            X_train_parts, y_train_parts = [], []
            for ds in train_dsets:
                g = load_graph(ds, tier="A", dev=dev)
                X_train_parts.append(g.edge_attr.numpy())
                y_train_parts.append(g.edge_label.numpy())
            X_train = np.concatenate(X_train_parts, axis=0)
            y_train = np.concatenate(y_train_parts, axis=0)

            g_test = load_graph(test_dset, tier="A", dev=dev)
            X_test = g_test.edge_attr.numpy()
            y_test = g_test.edge_label.numpy()

            rf = RandomForestBaseline(random_state=seed)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            save_model("B1_RF", seed, test_dset, rf.model)

            metrics = compute_all_metrics(
                y_test, y_pred,
                y_true_type=g_test.edge_label_type,
            )
            elapsed = time.time() - t0
            log_result("B1_RF", seed, train_dsets, test_dset, "mcc",      metrics["mcc"],      elapsed)
            log_result("B1_RF", seed, train_dsets, test_dset, "macro_f1", metrics["macro_f1"], elapsed)
            for cls, f1 in metrics.get("per_class_f1", {}).items():
                log_result("B1_RF", seed, train_dsets, test_dset, f"f1_{cls}", f1, elapsed)


def run_b4_b5(exp_id: str, use_quantile: bool, dev: bool = True):
    """B4 (raw Tier-A) or B5 (quantile Tier-A) — E-GraphSAGE LODO."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tier = "A"

    for fold_idx, fold in enumerate(FOLDS):
        train_dsets = fold["train"]
        test_dset   = fold["test"]
        for seed in SEEDS:
            if already_done(exp_id, seed, test_dset):
                log.info(f"Skipping {exp_id} seed={seed} test={test_dset} (already done)")
                continue
            seed_everything(seed)
            t0 = time.time()

            train_graphs = [load_graph(ds, tier=tier, dev=dev) for ds in train_dsets]
            combined = combine_graphs(train_graphs)
            test_graph = load_graph(test_dset, tier=tier, dev=dev)

            in_feats   = combined.edge_attr.shape[1]
            node_in    = combined.x.shape[1]
            val_split  = _lodo_val_split(combined, seed)
            model = EdgeAwareSAGE(node_in=node_in, edge_in=in_feats)
            best_state = train_egraphsage(
                model, combined, val_split=val_split, device=device,
                use_quantile=use_quantile,
            )
            model.load_state_dict(best_state)
            save_model(exp_id, seed, test_dset, best_state)

            result = eval_egraphsage(model, test_graph, device=device,
                                     use_quantile=use_quantile)
            metrics = compute_all_metrics(
                result["y_true"], result["y_pred"],
                y_true_type=test_graph.edge_label_type,
            )
            elapsed = time.time() - t0
            log_result(exp_id, seed, train_dsets, test_dset, "mcc",      metrics["mcc"],      elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "macro_f1", metrics["macro_f1"], elapsed)


def _lodo_val_split(combined, seed: int, val_frac: float = 0.2) -> dict:
    """Stratified random val split on a combined LODO training graph."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(combined.edge_label.shape[0])
    labels = combined.edge_label.numpy()
    ti, vi = train_test_split(idx, test_size=val_frac, random_state=seed, stratify=labels)
    return {"train": ti.tolist(), "val": vi.tolist()}


def _stratified_split(g, seed: int, val_frac: float = 0.2):
    """Stratified random split preserving attack/benign ratio in train and val."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(g.edge_label.shape[0])
    labels = g.edge_label.numpy()
    train_idx, val_idx = train_test_split(
        idx, test_size=val_frac, random_state=seed, stratify=labels
    )
    return {"train": train_idx.tolist(), "val": val_idx.tolist()}


def run_b3(dev: bool = True):
    """B3 — E-GraphSAGE within-dataset (upper bound, stratified split)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for ds in DATASETS:
        for seed in SEEDS:
            if already_done("B3_within", seed, ds):
                log.info(f"Skipping B3_within seed={seed} test={ds} (already done)")
                continue
            seed_everything(seed)
            t0 = time.time()

            g = load_graph(ds, tier="B", dev=dev)
            # Stratified split so train/val have same attack type mix (true upper bound)
            split = _stratified_split(g, seed=seed)

            model = EdgeAwareSAGE(node_in=g.x.shape[1], edge_in=g.edge_attr.shape[1])
            best_state = train_egraphsage(
                model, g, val_split=split, device=device, use_quantile=True,
            )
            model.load_state_dict(best_state)
            save_model("B3_within", seed, ds, best_state)

            result = eval_egraphsage(model, g, split["val"], device=device)
            metrics = compute_all_metrics(
                result["y_true"], result["y_pred"],
                y_true_type=[g.edge_label_type[i] for i in split["val"]],
            )
            elapsed = time.time() - t0
            log_result("B3_within", seed, [ds], ds, "mcc",      metrics["mcc"],      elapsed)
            log_result("B3_within", seed, [ds], ds, "macro_f1", metrics["macro_f1"], elapsed)


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", default=True)
    parser.add_argument("--no-dev", dest="dev", action="store_false")
    parser.add_argument("--baselines", nargs="+", default=["B1", "B3", "B4", "B5"])
    args = parser.parse_args()

    if "B1" in args.baselines:
        log.info("Running B1 — Random Forest")
        run_b1(dev=args.dev)

    if "B3" in args.baselines:
        log.info("Running B3 — E-GraphSAGE within-dataset")
        run_b3(dev=args.dev)

    if "B4" in args.baselines:
        log.info("Running B4 — E-GraphSAGE LODO raw")
        run_b4_b5("B4_raw", use_quantile=False, dev=args.dev)

    if "B5" in args.baselines:
        log.info("Running B5 — E-GraphSAGE LODO quantile")
        run_b4_b5("B5_quant", use_quantile=True, dev=args.dev)


if __name__ == "__main__":
    main()
