"""
Builds PyG Data objects from canonical DataFrames and saves them to disk.
Also produces temporal 80/20 splits and dev subsamples.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
SPLITS_DIR    = Path(__file__).resolve().parents[2] / "data" / "splits"
DEV_MAX_ROWS  = 2_000_000


def quantile_encode(x: torch.Tensor) -> torch.Tensor:
    """Per-graph quantile normalization: maps each feature to [0, 1]."""
    ranks = x.argsort(dim=0).argsort(dim=0).float()
    return ranks / (x.size(0) - 1 + 1e-8)


def _subsample_dev(df: pd.DataFrame, max_rows: int = DEV_MAX_ROWS) -> pd.DataFrame:
    """Temporally-stratified subsample preserving class ratios and timestamp order."""
    if len(df) <= max_rows:
        return df
    ratio = max_rows / len(df)
    sampled = (
        df.groupby("label_type", group_keys=False)
        .apply(lambda g: g.sample(frac=ratio, random_state=42), include_groups=False)
    )
    # include_groups=False drops label_type from the group frame; re-add it
    if "label_type" not in sampled.columns:
        sampled = sampled.join(df[["label_type"]])
    return sampled.sort_values("timestamp").reset_index(drop=True)


def build_graph(
    df: pd.DataFrame,
    feature_cols: List[str],
    dataset_name: str,
    tier: str = "B",
    save: bool = True,
    dev: bool = False,
) -> Data:
    """
    Build a PyG Data object from a canonical DataFrame.

    Args:
        df:           canonical DataFrame from a loader
        feature_cols: Tier-B feature column names (ignored when tier='A')
        dataset_name: e.g. 'lycos_ids2017'
        tier:         'A' (4-dim shared) or 'B' (full per-dataset)
        save:         write .pt files to data/processed/
        dev:          if True, subsample to DEV_MAX_ROWS first

    Returns:
        PyG Data with edge_index, edge_attr (raw), edge_attr_q (quantile),
        edge_time, edge_label, x (all-ones node features).
    """
    if dev:
        df = _subsample_dev(df)

    suffix = "_dev" if dev else "_full"

    # Feature matrix
    if tier == "A":
        from src.data.feature_aligner import extract_tier_a, TIER_A_COLS
        feat_df = extract_tier_a(df, dataset_name)
        used_cols = TIER_A_COLS
    else:
        from src.data.feature_aligner import extract_tier_b
        feat_df = extract_tier_b(df, feature_cols)
        used_cols = feature_cols

    edge_attr_np = feat_df.values.astype(np.float32)
    # Replace inf/nan with 0
    edge_attr_np = np.nan_to_num(edge_attr_np, nan=0.0, posinf=0.0, neginf=0.0)

    # Node mapping
    unique_ips = pd.unique(pd.concat([df["src_ip"], df["dst_ip"]]))
    ip_to_idx = {ip: i for i, ip in enumerate(unique_ips)}

    src_idx = df["src_ip"].map(ip_to_idx).values.astype(np.int64)
    dst_idx = df["dst_ip"].map(ip_to_idx).values.astype(np.int64)

    edge_index = torch.stack([
        torch.as_tensor(src_idx, dtype=torch.long),
        torch.as_tensor(dst_idx, dtype=torch.long),
    ])
    edge_attr  = torch.as_tensor(edge_attr_np, dtype=torch.float32)
    edge_time  = torch.as_tensor(df["timestamp"].values, dtype=torch.long)
    edge_label = torch.as_tensor(df["label_binary"].values, dtype=torch.long)
    edge_label_type = list(df["label_type"].values)
    x = torch.ones(len(unique_ips), 8, dtype=torch.float32)

    edge_attr_q = quantile_encode(edge_attr)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_attr_q=edge_attr_q,
        edge_time=edge_time,
        edge_label=edge_label,
    )
    data.edge_label_type = edge_label_type
    data.feature_cols    = used_cols
    data.num_nodes       = len(unique_ips)
    data.ip_to_idx       = ip_to_idx

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"{dataset_name}_tier{tier}{suffix}.pt"
        torch.save(data, PROCESSED_DIR / fname)
        log.info(f"Saved {fname}  edges={edge_index.shape[1]}  nodes={len(unique_ips)}")

        # Write feature manifest
        manifest_path = PROCESSED_DIR / f"{dataset_name}_manifest.json"
        manifest = {
            "dataset": dataset_name,
            "tier": tier,
            "feature_cols": used_cols,
            "num_features": len(used_cols),
            "num_edges_full": edge_index.shape[1] if not dev else None,
            "num_edges_dev":  edge_index.shape[1] if dev else None,
        }
        # merge with existing manifest if present
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            existing.update({k: v for k, v in manifest.items() if v is not None})
            manifest = existing
        manifest_path.write_text(json.dumps(manifest, indent=2))

        # Write temporal 80/20 split
        _save_temporal_split(dataset_name, edge_index.shape[1], df["timestamp"].values)

    return data


def _save_temporal_split(dataset_name: str, n_edges: int, timestamps: np.ndarray):
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split_idx = int(n_edges * 0.8)
    # find the actual index where timestamp crosses the 80th percentile
    thresh = np.percentile(timestamps, 80)
    train_mask = (timestamps <= thresh)
    train_idx = np.where(train_mask)[0].tolist()
    val_idx   = np.where(~train_mask)[0].tolist()
    split = {"train": train_idx, "val": val_idx, "split_timestamp": int(thresh)}
    out = SPLITS_DIR / f"{dataset_name}_temporal.json"
    out.write_text(json.dumps(split))
    log.info(f"Saved temporal split for {dataset_name}: train={len(train_idx)}, val={len(val_idx)}")


def load_graph(dataset_name: str, tier: str = "B", dev: bool = False) -> Data:
    suffix = "_dev" if dev else "_full"
    fname = PROCESSED_DIR / f"{dataset_name}_tier{tier}{suffix}.pt"
    if not fname.exists():
        raise FileNotFoundError(f"Graph not found: {fname}. Run scripts/preprocess.py first.")
    return torch.load(fname, weights_only=False)


def load_split(dataset_name: str):
    path = SPLITS_DIR / f"{dataset_name}_temporal.json"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}")
    return json.loads(path.read_text())


def combine_graphs(graphs: list, device: str = "cpu") -> Data:
    """
    Concatenate multiple graphs into one with disjoint node IDs.
    Used for LODO training: merge 3 training dataset graphs.
    """
    all_x, all_ei, all_ea, all_eaq, all_et, all_el, all_elt = [], [], [], [], [], [], []
    node_offset = 0
    max_feat = max(g.edge_attr.shape[1] for g in graphs)

    for g in graphs:
        n = g.num_nodes
        all_x.append(g.x)
        ei = g.edge_index + node_offset
        all_ei.append(ei)
        # zero-pad feature dim to max_feat
        ea = g.edge_attr
        if ea.shape[1] < max_feat:
            pad = torch.zeros(ea.shape[0], max_feat - ea.shape[1])
            ea = torch.cat([ea, pad], dim=1)
        eaq = g.edge_attr_q
        if eaq.shape[1] < max_feat:
            pad = torch.zeros(eaq.shape[0], max_feat - eaq.shape[1])
            eaq = torch.cat([eaq, pad], dim=1)
        all_ea.append(ea)
        all_eaq.append(eaq)
        all_et.append(g.edge_time)
        all_el.append(g.edge_label)
        all_elt.extend(g.edge_label_type)
        node_offset += n

    combined = Data(
        x          = torch.cat(all_x, dim=0),
        edge_index = torch.cat(all_ei, dim=1),
        edge_attr  = torch.cat(all_ea, dim=0),
        edge_attr_q= torch.cat(all_eaq, dim=0),
        edge_time  = torch.cat(all_et, dim=0),
        edge_label = torch.cat(all_el, dim=0),
    )
    combined.edge_label_type = all_elt
    combined.num_nodes = node_offset

    # sort by time for temporal models
    order = combined.edge_time.argsort()
    combined.edge_index  = combined.edge_index[:, order]
    combined.edge_attr   = combined.edge_attr[order]
    combined.edge_attr_q = combined.edge_attr_q[order]
    combined.edge_time   = combined.edge_time[order]
    combined.edge_label  = combined.edge_label[order]
    combined.edge_label_type = [combined.edge_label_type[i] for i in order.tolist()]

    return combined
