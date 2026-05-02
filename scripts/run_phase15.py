#!/usr/bin/env python3
"""
Phase 15: Calibration sweep for missing models.
  - E13.4_gib_nofeat_b0.01  seed 0  test=ton_iot
  - B4_raw                   seed 0  all 4 folds

For each checkpoint: runs test + val inference, computes all calibration
methods from Phase 13, prints a summary, and appends to calibration_v2.csv.

Usage:
    python scripts/run_phase15.py [--no-dev]
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split as _tts

from run_phase4 import ALL_FOLDS
from run_phase12 import _oracle_mcc
from run_phase13 import _all_cal_methods_v2
from src.data.graph_builder import combine_graphs, load_graph
from src.models.egraphsage import EdgeAwareSAGE
from src.models.gib_egraphsage import GIB_EGraphSAGE
from src.train.eval import eval_egraphsage
from src.utils.logging import MODELS_DIR, setup_logging

log = logging.getLogger(__name__)

CAL_V2_CSV = Path("results/calibration_v2.csv")

# (exp_id, seed, test_fold, tier, use_quantile, nofeat)
# nofeat=True → replace all edge features with ones before inference
TARGETS = [
    ("E13.4_gib_nofeat_b0.01", 0, "ton_iot",       "B", True,  True),
    ("B4_raw",                  0, "lycos_ids2017",  "A", False, False),
    ("B4_raw",                  0, "cic_ids2018",    "A", False, False),
    ("B4_raw",                  0, "unsw_nb15",      "A", False, False),
    ("B4_raw",                  0, "ton_iot",        "A", False, False),
]

CAL_METHODS = [
    "p11_topk_src_rate", "p11_otsu", "p11_gmm",
    "p11_topk_10pct", "p11_topk_20pct", "p11_topk_30pct",
    "val_anchor", "znorm_val", "platt", "bbse", "gmm_logit", "ensemble",
]
FIELDNAMES = [
    "experiment_id", "seed", "test_fold",
    "p11_reported_0.5", "p11_topk_src_rate", "p11_otsu",
    "p11_gmm", "p11_topk_10pct", "p11_topk_20pct", "p11_topk_30pct",
    "p11_oracle",
    "val_anchor", "znorm_val", "platt", "bbse", "gmm_logit", "ensemble",
    "oracle_mcc", "auroc",
]


def _ones(g):
    """Replace edge features with ones in-place (structure-only models)."""
    import copy
    g = copy.copy(g)
    E = g.edge_attr.shape[0]
    g.edge_attr   = torch.ones(E, 1)
    g.edge_attr_q = torch.ones(E, 1)
    return g


@torch.no_grad()
def _run_target(exp_id, seed, test_fold, tier, use_quantile, nofeat, dev, device):
    """
    Load checkpoint, run test+val inference, return (test_scores, test_labels,
    val_scores, val_labels, p_src). Returns None on failure.
    """
    ckpt = MODELS_DIR / f"{exp_id}_seed{seed}_test{test_fold}.pt"
    if not ckpt.exists():
        log.warning(f"  Checkpoint not found: {ckpt.name}")
        return None

    fold = next((f for f in ALL_FOLDS if f["test"] == test_fold), None)
    if fold is None:
        log.error(f"  Unknown fold: {test_fold}")
        return None

    state = torch.load(ckpt, weights_only=True)
    is_gib = any("to_dist" in k for k in state)
    ck_key = ("encoder.edge_enc.0.weight"
              if "encoder.edge_enc.0.weight" in state
              else "edge_enc.0.weight")
    ck_edge_in = state[ck_key].shape[1]

    # Load graphs
    train_graphs = [load_graph(ds, tier=tier, dev=dev) for ds in fold["train"]]
    if nofeat:
        train_graphs = [_ones(g) for g in train_graphs]
    combined = combine_graphs(train_graphs)

    test_graph = load_graph(test_fold, tier=tier, dev=dev)
    if nofeat:
        test_graph = _ones(test_graph)

    # Build model
    ck_node_in = combined.x.shape[1]
    if is_gib:
        model = GIB_EGraphSAGE(node_in=ck_node_in, edge_in=ck_edge_in)
    else:
        model = EdgeAwareSAGE(node_in=ck_node_in, edge_in=ck_edge_in)
    model.load_state_dict(state)
    model.eval().to(device)

    # Test inference
    result      = eval_egraphsage(model, test_graph, device=device, use_quantile=use_quantile)
    test_scores = result["y_score"].astype(np.float32)
    test_labels = result["y_true"]

    # Reproduce val split (stratified 80/20, same for both model families)
    labels_np = combined.edge_label.numpy()
    _, vi = _tts(np.arange(len(labels_np)), test_size=0.2,
                 random_state=seed, stratify=labels_np)
    vi = np.array(vi, dtype=np.int64)

    x  = combined.x.to(device)
    ei = combined.edge_index.to(device)
    ea = (combined.edge_attr_q if use_quantile else combined.edge_attr).to(device)
    if ea.shape[1] != ck_edge_in:
        ea = (ea[:, :ck_edge_in] if ea.shape[1] > ck_edge_in
              else torch.cat([ea, torch.zeros(ea.shape[0], ck_edge_in - ea.shape[1],
                                              device=device)], dim=1))

    val_parts = []
    for s in range(0, len(vi), 50_000):
        idx    = vi[s:s + 50_000]
        logits = model(x, ei[:, idx], ea[idx])
        val_parts.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    val_scores = np.concatenate(val_parts).astype(np.float32)
    val_labels = labels_np[vi]

    p_src = float(val_labels.mean())
    return test_scores, test_labels, val_scores, val_labels, p_src


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-dev", dest="dev", action="store_false")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Phase 15 — calibration sweep  dev={args.dev}  device={device}")

    # Read already-done rows
    done_keys = set()
    CAL_V2_CSV.parent.mkdir(parents=True, exist_ok=True)
    if CAL_V2_CSV.exists():
        with open(CAL_V2_CSV) as f:
            for row in csv.DictReader(f):
                done_keys.add((row["experiment_id"], row["seed"], row["test_fold"]))

    with open(CAL_V2_CSV, "a", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not done_keys:
            writer.writeheader()

        summary_rows = []

        for exp_id, seed, test_fold, tier, use_q, nofeat in TARGETS:
            key = (exp_id, str(seed), test_fold)
            if key in done_keys:
                log.info(f"  Skip {exp_id} seed={seed} fold={test_fold} (already in csv)")
                continue

            log.info(f"\n  {exp_id}  seed={seed}  fold={test_fold}")
            out = _run_target(exp_id, seed, test_fold, tier, use_q, nofeat,
                              args.dev, device)
            if out is None:
                continue
            test_scores, test_labels, val_scores, val_labels, p_src = out

            cal = _all_cal_methods_v2(val_scores, val_labels, test_scores, test_labels, p_src)
            orc, _ = _oracle_mcc(test_scores, test_labels)
            try:
                auroc = float(roc_auc_score(test_labels, test_scores))
            except Exception:
                auroc = float("nan")

            log.info(f"  p_src={p_src:.4f}  oracle={orc:.4f}  auroc={auroc:.4f}")
            best_m, best_v = max(
                ((m, cal.get(m, float("nan"))) for m in CAL_METHODS
                 if not np.isnan(cal.get(m, float("nan")))),
                key=lambda x: x[1], default=("—", float("nan")),
            )
            log.info(f"  best method: {best_m} = {best_v:.4f}")

            row_out = {"experiment_id": exp_id, "seed": str(seed),
                       "test_fold": test_fold,
                       "oracle_mcc": f"{orc:.6f}", "auroc": f"{auroc:.6f}"}
            for k, v in cal.items():
                row_out[k] = f"{v:.6f}" if not np.isnan(v) else "nan"
            writer.writerow(row_out)
            f_out.flush()
            done_keys.add(key)

            summary_rows.append((exp_id, test_fold, best_m, best_v, orc, auroc))

    # Summary table
    if summary_rows:
        log.info(f"\n{'='*65}")
        log.info("  Phase 15 summary")
        log.info(f"  {'exp_id':<32} {'fold':<18} {'best_method':<22} {'best_mcc':>8} {'oracle':>8} {'auroc':>7}")
        for exp_id, fold, bm, bv, orc, auroc in summary_rows:
            log.info(f"  {exp_id:<32} {fold:<18} {bm:<22}"
                     f"  {bv:>8.4f}  {orc:>8.4f}  {auroc:>7.4f}")


if __name__ == "__main__":
    main()
