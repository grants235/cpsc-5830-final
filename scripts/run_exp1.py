#!/usr/bin/env python3
"""
Experiment 1 — Relative Feature Encoding (RFE)
E1.A: raw Tier-A           = B4 reference
E1.B: quantile Tier-A      = B5 reference
E1.C: quantile Tier-B      (main RFE result)
E1.D: raw Tier-B, per-dataset encoder heads
E1.E: quantile Tier-B, structure-only (edge_attr masked to 1.0)

Usage:
    python scripts/run_exp1.py [--dev] [--exps E1.C E1.E]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np

from src.utils.logging import setup_logging, log_result, already_done, save_model
from src.utils.seeding import seed_everything
from sklearn.model_selection import train_test_split as _tts
from src.utils.metrics import compute_all_metrics
from src.data.graph_builder import load_graph, combine_graphs, load_split
from src.models.egraphsage import EdgeAwareSAGE
from src.train.train_loops import train_egraphsage
from src.train.eval import eval_egraphsage

log = logging.getLogger(__name__)

FOLDS = [
    {"train": ["cic_ids2018", "unsw_nb15", "ton_iot"],       "test": "lycos_ids2017"},
    {"train": ["lycos_ids2017", "unsw_nb15", "ton_iot"],     "test": "cic_ids2018"},
    {"train": ["lycos_ids2017", "cic_ids2018", "ton_iot"],   "test": "unsw_nb15"},
    {"train": ["lycos_ids2017", "cic_ids2018", "unsw_nb15"], "test": "ton_iot"},
]
SEEDS = [0, 1, 2]


def _run_standard(exp_id, tier, use_quantile, dev, structure_only=False):
    """Single encoder shared across training datasets."""
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

            train_graphs = [load_graph(ds, tier=tier, dev=dev) for ds in train_dsets]
            combined     = combine_graphs(train_graphs)
            test_graph   = load_graph(test_dset, tier=tier, dev=dev)

            # Align test graph feature dim to combined training dim
            max_feat = combined.edge_attr.shape[1]
            d = test_graph.edge_attr.shape[1]
            if d < max_feat:
                pad = torch.zeros(test_graph.edge_attr.shape[0], max_feat - d)
                test_graph.edge_attr   = torch.cat([test_graph.edge_attr, pad], dim=1)
                test_graph.edge_attr_q = torch.cat([test_graph.edge_attr_q, pad], dim=1)
            elif d > max_feat:
                test_graph.edge_attr   = test_graph.edge_attr[:, :max_feat]
                test_graph.edge_attr_q = test_graph.edge_attr_q[:, :max_feat]

            if structure_only:
                # Mask all edge features to 1.0
                combined.edge_attr   = torch.ones_like(combined.edge_attr)
                combined.edge_attr_q = torch.ones_like(combined.edge_attr_q)
                test_graph.edge_attr   = torch.ones_like(test_graph.edge_attr)
                test_graph.edge_attr_q = torch.ones_like(test_graph.edge_attr_q)

            n = combined.edge_label.shape[0]
            ti, vi = _tts(np.arange(n), test_size=0.2, random_state=seed,
                          stratify=combined.edge_label.numpy())
            val_split = {"train": ti.tolist(), "val": vi.tolist()}
            model = EdgeAwareSAGE(
                node_in=combined.x.shape[1],
                edge_in=combined.edge_attr.shape[1],
            )
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
            for cls, f1 in metrics.get("per_class_f1", {}).items():
                log_result(exp_id, seed, train_dsets, test_dset, f"f1_{cls}", f1, elapsed)


def _run_e1d(dev: bool):
    """E1.D: raw Tier-B, per-dataset encoder, mean-pool embeddings at test."""
    from src.models.baseline_mlp import EnsembleMLP
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold in FOLDS:
        train_dsets = fold["train"]
        test_dset   = fold["test"]
        for seed in SEEDS:
            if already_done("E1.D", seed, test_dset):
                log.info(f"Skipping E1.D seed={seed} test={test_dset} (already done)")
                continue
            seed_everything(seed)
            t0 = time.time()

            train_graphs = [load_graph(ds, tier="B", dev=dev) for ds in train_dsets]
            test_graph   = load_graph(test_dset, tier="B", dev=dev)

            feat_dims = [g.edge_attr.shape[1] for g in train_graphs]
            model = EnsembleMLP(feat_dims=feat_dims).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
            criterion = torch.nn.CrossEntropyLoss()

            for g_idx, g in enumerate(train_graphs):
                X = g.edge_attr.to(device)
                y = g.edge_label.to(device)
                for epoch in range(10):
                    optimizer.zero_grad()
                    logits = model.forward_single(X, g_idx)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()

            # Test: ensemble all encoders using zero-padded test features
            model.eval()
            max_feat = max(feat_dims)
            test_X = test_graph.edge_attr.to(device)
            xs = []
            for d in feat_dims:
                if d <= test_X.shape[1]:
                    xs.append(test_X[:, :d])
                else:
                    pad = torch.zeros(test_X.shape[0], d - test_X.shape[1], device=device)
                    xs.append(torch.cat([test_X, pad], dim=1))

            with torch.no_grad():
                logits = model.forward_ensemble(xs)
            y_pred = logits.argmax(dim=-1).cpu().numpy()
            y_true = test_graph.edge_label.numpy()
            save_model("E1.D", seed, test_dset, model.state_dict())

            metrics = compute_all_metrics(y_true, y_pred,
                                          y_true_type=test_graph.edge_label_type)
            elapsed = time.time() - t0
            log_result("E1.D", seed, train_dsets, test_dset, "mcc",      metrics["mcc"],      elapsed)
            log_result("E1.D", seed, train_dsets, test_dset, "macro_f1", metrics["macro_f1"], elapsed)


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", default=True)
    parser.add_argument("--no-dev", dest="dev", action="store_false")
    parser.add_argument("--exps", nargs="+",
                        default=["E1.A", "E1.B", "E1.C", "E1.D", "E1.E"])
    args = parser.parse_args()

    if "E1.A" in args.exps:
        log.info("Running E1.A — raw Tier-A (= B4)")
        _run_standard("E1.A", tier="A", use_quantile=False, dev=args.dev)

    if "E1.B" in args.exps:
        log.info("Running E1.B — quantile Tier-A (= B5)")
        _run_standard("E1.B", tier="A", use_quantile=True, dev=args.dev)

    if "E1.C" in args.exps:
        log.info("Running E1.C — quantile Tier-B (main RFE)")
        _run_standard("E1.C", tier="B", use_quantile=True, dev=args.dev)

    if "E1.D" in args.exps:
        log.info("Running E1.D — raw Tier-B per-dataset encoders")
        _run_e1d(dev=args.dev)

    if "E1.E" in args.exps:
        log.info("Running E1.E — structure only (features masked to 1.0)")
        _run_standard("E1.E", tier="B", use_quantile=True, dev=args.dev,
                      structure_only=True)


if __name__ == "__main__":
    main()
