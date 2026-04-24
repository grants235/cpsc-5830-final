#!/usr/bin/env python3
"""
Experiment 3 — Mixture of Experts (MoE / GraphMETRO-inspired)
E3.A: reference only  (= E1.C re-run)
E3.B: full 3-expert MoE
E3.C: MoE with uniform gating
E3.D: 3 independent E-GraphSAGEs ensembled (no alignment loss)
E3.E: E3.B without L_align

Usage:
    python scripts/run_exp3.py [--dev] [--exps E3.B E3.C]
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
from sklearn.model_selection import train_test_split as _tts
from src.utils.metrics import compute_all_metrics
from src.data.graph_builder import load_graph, combine_graphs
from src.models.moe_ids import MoE_IDS
from src.models.egraphsage import EdgeAwareSAGE
from src.train.train_loops import train_moe, train_egraphsage
from src.train.eval import eval_egraphsage

log = logging.getLogger(__name__)

FOLDS = [
    {"train": ["cic_ids2018", "unsw_nb15", "ton_iot"],       "test": "lycos_ids2017"},
    {"train": ["lycos_ids2017", "unsw_nb15", "ton_iot"],     "test": "cic_ids2018"},
    {"train": ["lycos_ids2017", "cic_ids2018", "ton_iot"],   "test": "unsw_nb15"},
    {"train": ["lycos_ids2017", "cic_ids2018", "unsw_nb15"], "test": "ton_iot"},
]
SEEDS = [0, 1, 2]


def _run_moe(exp_id: str, dev: bool, uniform_gate: bool = False, no_align: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lam = 0.0 if no_align else 0.1

    for fold in FOLDS:
        train_dsets = fold["train"]
        test_dset   = fold["test"]
        for seed in SEEDS:
            if already_done(exp_id, seed, test_dset):
                log.info(f"Skipping {exp_id} seed={seed} test={test_dset} (already done)")
                continue
            seed_everything(seed)
            t0 = time.time()

            train_graphs = [load_graph(ds, tier="B", dev=dev) for ds in train_dsets]
            combined     = combine_graphs(train_graphs)
            test_graph   = load_graph(test_dset, tier="B", dev=dev)

            max_feat = combined.edge_attr.shape[1]
            if test_graph.edge_attr.shape[1] < max_feat:
                pad = torch.zeros(test_graph.edge_attr.shape[0],
                                  max_feat - test_graph.edge_attr.shape[1])
                test_graph.edge_attr   = torch.cat([test_graph.edge_attr, pad], dim=1)
                test_graph.edge_attr_q = torch.cat([test_graph.edge_attr_q, pad], dim=1)

            n = combined.edge_label.shape[0]
            ti, vi = _tts(np.arange(n), test_size=0.2, random_state=seed,
                          stratify=combined.edge_label.numpy())
            val_split = {"train": ti.tolist(), "val": vi.tolist()}
            model = MoE_IDS(
                node_in=combined.x.shape[1],
                edge_in=max_feat,
                lam=lam,
            )
            best_state = train_moe(
                model, combined, val_split=val_split, device=device,
                uniform_gate=uniform_gate, lam=lam,
            )
            model.load_state_dict(best_state)
            save_model(exp_id, seed, test_dset, best_state)
            model.eval().to(device)

            x  = test_graph.x.to(device)
            ei = test_graph.edge_index.to(device)
            ea = test_graph.edge_attr_q.to(device)
            with torch.no_grad():
                logits = model(x, ei, ea)
            y_pred = logits.argmax(dim=-1).cpu().numpy()
            y_true = test_graph.edge_label.numpy()

            metrics = compute_all_metrics(y_true, y_pred,
                                          y_true_type=test_graph.edge_label_type)
            elapsed = time.time() - t0
            log_result(exp_id, seed, train_dsets, test_dset, "mcc",      metrics["mcc"],      elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "macro_f1", metrics["macro_f1"], elapsed)
            for cls, f1 in metrics.get("per_class_f1", {}).items():
                log_result(exp_id, seed, train_dsets, test_dset, f"f1_{cls}", f1, elapsed)


def _run_ensemble(exp_id: str, dev: bool):
    """E3.D: 3 independent E-GraphSAGEs, mean-pool logits at test (no alignment)."""
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

            train_graphs = [load_graph(ds, tier="B", dev=dev) for ds in train_dsets]
            test_graph   = load_graph(test_dset, tier="B", dev=dev)
            max_feat     = max(g.edge_attr.shape[1] for g in train_graphs)

            models = []
            for i, g in enumerate(train_graphs):
                model = EdgeAwareSAGE(node_in=g.x.shape[1], edge_in=g.edge_attr.shape[1])
                best  = train_egraphsage(model, g, val_split=None, device=device)
                model.load_state_dict(best)
                models.append(model)

            # Pad test features for each model's feature dimension
            test_logits = []
            for i, (model, g) in enumerate(zip(models, train_graphs)):
                model.eval().to(device)
                d = g.edge_attr.shape[1]
                x  = test_graph.x.to(device)
                ei = test_graph.edge_index.to(device)
                ea = test_graph.edge_attr_q.to(device)
                if ea.shape[1] < d:
                    pad = torch.zeros(ea.shape[0], d - ea.shape[1], device=device)
                    ea = torch.cat([ea, pad], dim=1)
                elif ea.shape[1] > d:
                    ea = ea[:, :d]
                with torch.no_grad():
                    test_logits.append(model(x, ei, ea))

            logits_mean = torch.stack(test_logits, dim=0).mean(dim=0)
            y_pred = logits_mean.argmax(dim=-1).cpu().numpy()
            y_true = test_graph.edge_label.numpy()
            save_model(exp_id, seed, test_dset,
                       [m.state_dict() for m in models])

            metrics = compute_all_metrics(y_true, y_pred,
                                          y_true_type=test_graph.edge_label_type)
            elapsed = time.time() - t0
            log_result(exp_id, seed, train_dsets, test_dset, "mcc",      metrics["mcc"],      elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "macro_f1", metrics["macro_f1"], elapsed)


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", default=True)
    parser.add_argument("--no-dev", dest="dev", action="store_false")
    parser.add_argument("--exps", nargs="+",
                        default=["E3.A", "E3.B", "E3.C", "E3.D", "E3.E"])
    args = parser.parse_args()

    if "E3.A" in args.exps:
        log.info("Running E3.A — reference only (= E1.C)")
        from scripts.run_exp1 import _run_standard
        _run_standard("E3.A", tier="B", use_quantile=True, dev=args.dev)

    if "E3.B" in args.exps:
        log.info("Running E3.B — full 3-expert MoE")
        _run_moe("E3.B", dev=args.dev)

    if "E3.C" in args.exps:
        log.info("Running E3.C — MoE uniform gating")
        _run_moe("E3.C", dev=args.dev, uniform_gate=True)

    if "E3.D" in args.exps:
        log.info("Running E3.D — independent ensemble (no alignment)")
        _run_ensemble("E3.D", dev=args.dev)

    if "E3.E" in args.exps:
        log.info("Running E3.E — MoE without L_align")
        _run_moe("E3.E", dev=args.dev, no_align=True)


if __name__ == "__main__":
    main()
