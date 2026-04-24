#!/usr/bin/env python3
"""
One-shot preprocessing: load all datasets, canonicalize, build PyG graphs,
save .pt files to data/processed/.

Usage:
    python scripts/preprocess.py [--dev] [--datasets lycos_ids2017 cic_ids2018 ...]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import setup_logging
from src.data.loaders import LOADERS
from src.data.graph_builder import build_graph

log = logging.getLogger(__name__)

ALL_DATASETS = ["lycos_ids2017", "cic_ids2018", "unsw_nb15", "ton_iot"]


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true",
                        help="Also build _dev (≤2M) subsampled graphs")
    parser.add_argument("--full-only", action="store_true",
                        help="Build full graphs only (no dev)")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    parser.add_argument("--tier", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    tiers = {"A": ["A"], "B": ["B"], "both": ["A", "B"]}[args.tier]

    for ds in args.datasets:
        log.info(f"=== {ds} ===")
        loader = LOADERS[ds]
        df, feat_cols = loader()
        log.info(f"  Loaded {len(df):,} rows")

        for tier in tiers:
            log.info(f"  Building Tier-{tier} full graph...")
            build_graph(df, feat_cols, ds, tier=tier, save=True, dev=False)

            if args.dev and not args.full_only:
                log.info(f"  Building Tier-{tier} dev graph...")
                build_graph(df, feat_cols, ds, tier=tier, save=True, dev=True)

    log.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
