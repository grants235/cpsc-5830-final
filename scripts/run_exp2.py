#!/usr/bin/env python3
"""
Experiment 2 — Temporal Graph Networks (TGN)
E2.A: quantile Tier-B, no TGN   (= E1.C re-run)
E2.B: quantile Tier-B, TGN full  (main temporal result)
E2.C: raw Tier-B, TGN full
E2.D: quantile Tier-B, TGN memory-only (no graph attention)
E2.E: quantile Tier-B, TGN with memory reset between training datasets

Usage:
    python scripts/run_exp2.py [--dev] [--exps E2.B E2.D]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.utils.logging import setup_logging, log_result, already_done, save_model
from src.utils.seeding import seed_everything
from src.utils.metrics import compute_all_metrics
from src.data.graph_builder import load_graph, combine_graphs
from src.models.tgn_ids import TGN_IDS, TGN_MemoryOnly
from src.train.train_loops import train_tgn
from src.train.eval import eval_tgn
from torch_geometric.nn.models.tgn import LastNeighborLoader

log = logging.getLogger(__name__)

FOLDS = [
    {"train": ["cic_ids2018", "unsw_nb15", "ton_iot"],       "test": "lycos_ids2017"},
    {"train": ["lycos_ids2017", "unsw_nb15", "ton_iot"],     "test": "cic_ids2018"},
    {"train": ["lycos_ids2017", "cic_ids2018", "ton_iot"],   "test": "unsw_nb15"},
    {"train": ["lycos_ids2017", "cic_ids2018", "unsw_nb15"], "test": "ton_iot"},
]
SEEDS = [0, 1, 2]


def _dataset_boundaries(train_graphs: list) -> list:
    """Edge indices where a new training dataset begins (for memory reset ablation)."""
    boundaries = []
    offset = 0
    for g in train_graphs[:-1]:
        offset += g.edge_index.shape[1]
        boundaries.append(offset)
    return boundaries


def _run_tgn(exp_id: str, use_quantile: bool, dev: bool,
             memory_only: bool = False, reset_between: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold in FOLDS:
        train_dsets = fold["train"]
        test_dset   = fold["test"]
        for seed in SEEDS:
            if already_done(exp_id, seed, test_dset):
                log.info(f"Skipping {exp_id} seed={seed} test={test_dset} (already done)")
                continue
            seed_everything(seed)
            t0 = time.time()

            train_graphs  = [load_graph(ds, tier="B", dev=dev) for ds in train_dsets]
            boundaries    = _dataset_boundaries(train_graphs)
            combined      = combine_graphs(train_graphs)
            test_graph    = load_graph(test_dset, tier="B", dev=dev)

            # Pad test graph features to match combined if needed
            max_feat = combined.edge_attr.shape[1]
            if test_graph.edge_attr.shape[1] < max_feat:
                pad = torch.zeros(test_graph.edge_attr.shape[0],
                                  max_feat - test_graph.edge_attr.shape[1])
                test_graph.edge_attr   = torch.cat([test_graph.edge_attr, pad], dim=1)
                test_graph.edge_attr_q = torch.cat([test_graph.edge_attr_q, pad], dim=1)

            raw_msg_dim = max_feat
            num_nodes   = combined.num_nodes + test_graph.num_nodes  # safe upper bound

            if memory_only:
                model = TGN_MemoryOnly(num_nodes, raw_msg_dim)
            else:
                model = TGN_IDS(num_nodes, raw_msg_dim)

            best_state = train_tgn(
                model, combined, val_data=None, device=device,
                use_quantile=use_quantile,
                reset_memory_between_datasets=reset_between,
                dataset_boundaries=boundaries if reset_between else None,
            )
            model.load_state_dict(best_state)
            save_model(exp_id, seed, test_dset, best_state)

            neighbor_loader = LastNeighborLoader(num_nodes, size=10, device=device)
            assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
            result = eval_tgn(model, test_graph, neighbor_loader, assoc,
                              device=device, use_quantile=use_quantile)

            metrics = compute_all_metrics(
                result["y_true"], result["y_pred"],
                y_true_type=test_graph.edge_label_type,
            )
            elapsed = time.time() - t0
            log_result(exp_id, seed, train_dsets, test_dset, "mcc",      metrics["mcc"],      elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "macro_f1", metrics["macro_f1"], elapsed)
            for cls, f1 in metrics.get("per_class_f1", {}).items():
                log_result(exp_id, seed, train_dsets, test_dset, f"f1_{cls}", f1, elapsed)


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", default=True)
    parser.add_argument("--no-dev", dest="dev", action="store_false")
    parser.add_argument("--exps", nargs="+",
                        default=["E2.B", "E2.C", "E2.D", "E2.E"])
    args = parser.parse_args()

    if "E2.A" in args.exps:
        log.info("Running E2.A — quantile Tier-B no TGN (= E1.C)")
        from scripts.run_exp1 import _run_standard
        _run_standard("E2.A", tier="B", use_quantile=True, dev=args.dev)

    if "E2.B" in args.exps:
        log.info("Running E2.B — TGN full, quantile Tier-B")
        _run_tgn("E2.B", use_quantile=True, dev=args.dev)

    if "E2.C" in args.exps:
        log.info("Running E2.C — TGN full, raw Tier-B")
        _run_tgn("E2.C", use_quantile=False, dev=args.dev)

    if "E2.D" in args.exps:
        log.info("Running E2.D — TGN memory-only")
        _run_tgn("E2.D", use_quantile=True, dev=args.dev, memory_only=True)

    if "E2.E" in args.exps:
        log.info("Running E2.E — TGN with memory reset between datasets")
        _run_tgn("E2.E", use_quantile=True, dev=args.dev, reset_between=True)


if __name__ == "__main__":
    main()
