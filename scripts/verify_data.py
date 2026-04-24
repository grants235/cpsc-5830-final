#!/usr/bin/env python3
"""
Sanity checks on processed graphs (§2.8).
Must pass before running any experiment.

Usage:
    python scripts/verify_data.py [--dev]
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.utils.logging import setup_logging
from src.data.graph_builder import load_graph, PROCESSED_DIR

log = logging.getLogger(__name__)

DATASETS = ["lycos_ids2017", "cic_ids2018", "unsw_nb15", "ton_iot"]

# Expected minimum edge counts for *_full (UNSW-NB15 is small ~2.5M with all 4 files)
MIN_EDGES = {
    "lycos_ids2017": 1_000_000,
    "cic_ids2018":   5_000_000,
    "unsw_nb15":     100_000,   # smaller dataset, relax threshold
    "ton_iot":       5_000_000,
}


def check_graph(ds: str, tier: str = "B", dev: bool = False):
    suffix = "dev" if dev else "full"
    try:
        g = load_graph(ds, tier=tier, dev=dev)
    except FileNotFoundError as e:
        log.error(f"MISSING: {e}")
        return False

    ok = True
    E = g.edge_index.shape[1]
    F = g.edge_attr.shape[1]

    # 1. Edge count
    if not dev:
        min_e = MIN_EDGES.get(ds, 100_000)
        if E < min_e:
            log.error(f"[{ds}/{suffix}] edge count {E:,} < {min_e:,}")
            ok = False
        else:
            log.info(f"[{ds}/{suffix}] edges={E:,} ✓")

    # 2. No NaN / Inf in edge_attr
    if torch.isnan(g.edge_attr).any() or torch.isinf(g.edge_attr).any():
        log.error(f"[{ds}/{suffix}] NaN/Inf in edge_attr!")
        ok = False
    else:
        log.info(f"[{ds}/{suffix}] edge_attr clean ✓")

    if torch.isnan(g.edge_attr_q).any() or torch.isinf(g.edge_attr_q).any():
        log.error(f"[{ds}/{suffix}] NaN/Inf in edge_attr_q!")
        ok = False
    else:
        log.info(f"[{ds}/{suffix}] edge_attr_q clean ✓")

    # 3. Timestamps monotonically non-decreasing
    ts = g.edge_time
    if (ts[1:] < ts[:-1]).any():
        log.error(f"[{ds}/{suffix}] timestamps NOT monotonic!")
        ok = False
    else:
        log.info(f"[{ds}/{suffix}] timestamps monotonic ✓")

    # 4. Feature count matches manifest
    manifest_path = PROCESSED_DIR / f"{ds}_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected_F = manifest.get("num_features")
        if expected_F and F != expected_F:
            log.error(f"[{ds}/{suffix}] feature dim {F} != manifest {expected_F}")
            ok = False
        else:
            log.info(f"[{ds}/{suffix}] feature dim={F} matches manifest ✓")

    # 5. Label distribution check
    n_attack = g.edge_label.sum().item()
    n_total  = E
    pct_attack = n_attack / n_total * 100
    log.info(f"[{ds}/{suffix}] attack%={pct_attack:.1f}% ({n_attack:,}/{n_total:,})")

    # 6. Quantile features in [0, 1]
    if g.edge_attr_q.min() < -1e-3 or g.edge_attr_q.max() > 1 + 1e-3:
        log.error(f"[{ds}/{suffix}] edge_attr_q out of [0,1] range!")
        ok = False
    else:
        log.info(f"[{ds}/{suffix}] edge_attr_q in [0,1] ✓")

    return ok


def main():
    import argparse
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--tier", default="B")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()

    all_ok = True
    for ds in args.datasets:
        ok = check_graph(ds, tier=args.tier, dev=False)
        all_ok = all_ok and ok
        if args.dev:
            ok_dev = check_graph(ds, tier=args.tier, dev=True)
            all_ok = all_ok and ok_dev

    if all_ok:
        log.info("All checks PASSED ✓")
    else:
        log.error("Some checks FAILED ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
