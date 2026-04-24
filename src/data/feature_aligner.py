"""
Tier-A shared feature extraction (4 features common across all datasets)
and Tier-B full per-dataset feature set.

Tier-A columns produced:
  byte_count        total bytes (src+dst)
  packet_count      total packets (src+dst)
  tcp_flags_any     bitwise OR of all flag columns (0 for non-TCP)
  flow_duration_ms  flow duration in milliseconds
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Tier-A extraction per dataset
# ---------------------------------------------------------------------------

def _tier_a_lycos(df: pd.DataFrame) -> pd.DataFrame:
    byte_count = (
        df.get("fwd_pkt_len_tot", 0) + df.get("bwd_pkt_len_tot", 0)
    ).astype("float32")
    pkt_count = (
        df.get("fwd_pkt_cnt", 0) + df.get("bwd_pkt_cnt", 0)
    ).astype("float32")
    flag_cols = ["flag_SYN", "flag_fin", "flag_rst", "flag_ack",
                 "flag_psh", "flag_urg", "flag_cwr", "flag_ece"]
    # treat flag counts as binary presence
    flags = np.zeros(len(df), dtype=np.int32)
    for bit, col in enumerate(flag_cols):
        if col in df.columns:
            presence = (df[col].fillna(0).values.astype(np.int32) > 0).astype(np.int32)
            flags |= (presence << bit)
    dur_ms = df.get("flow_duration", pd.Series(0, index=df.index)).astype("float32")
    out = pd.DataFrame({
        "byte_count":       byte_count,
        "packet_count":     pkt_count,
        "tcp_flags_any":    flags.astype("float32"),
        "flow_duration_ms": dur_ms,
    }, index=df.index)
    return out


def _tier_a_cic18(df: pd.DataFrame) -> pd.DataFrame:
    byte_count = (df["IN_BYTES"] + df["OUT_BYTES"]).astype("float32")
    pkt_count  = (df["IN_PKTS"]  + df["OUT_PKTS"]).astype("float32")
    flags      = df["TCP_FLAGS"].fillna(0).astype("float32")
    dur_ms     = df["FLOW_DURATION_MILLISECONDS"].astype("float32")
    return pd.DataFrame({
        "byte_count":       byte_count,
        "packet_count":     pkt_count,
        "tcp_flags_any":    flags,
        "flow_duration_ms": dur_ms,
    }, index=df.index)


def _tier_a_unsw(df: pd.DataFrame) -> pd.DataFrame:
    byte_count = (
        pd.to_numeric(df["sbytes"], errors="coerce").fillna(0) +
        pd.to_numeric(df["dbytes"], errors="coerce").fillna(0)
    ).astype("float32")
    pkt_count = (
        pd.to_numeric(df["Spkts"], errors="coerce").fillna(0) +
        pd.to_numeric(df["Dpkts"], errors="coerce").fillna(0)
    ).astype("float32")
    # proto dropped from canonical df — use swin>0 as TCP proxy (swin only set for TCP)
    if "proto" in df.columns:
        is_tcp = (df["proto"].astype(str).str.lower() == "tcp").astype("float32")
    elif "swin" in df.columns:
        is_tcp = (pd.to_numeric(df["swin"], errors="coerce").fillna(0) > 0).astype("float32")
    else:
        is_tcp = pd.Series(0.0, index=df.index, dtype="float32")
    dur_ms = (
        pd.to_numeric(df["dur"], errors="coerce").fillna(0) * 1000
    ).astype("float32")
    return pd.DataFrame({
        "byte_count":       byte_count,
        "packet_count":     pkt_count,
        "tcp_flags_any":    is_tcp,
        "flow_duration_ms": dur_ms,
    }, index=df.index)


def _tier_a_ton(df: pd.DataFrame) -> pd.DataFrame:
    byte_count = (df["IN_BYTES"] + df["OUT_BYTES"]).astype("float32")
    pkt_count  = (df["IN_PKTS"]  + df["OUT_PKTS"]).astype("float32")
    flags      = df["TCP_FLAGS"].fillna(0).astype("float32")
    dur_ms     = df["FLOW_DURATION_MILLISECONDS"].astype("float32")
    return pd.DataFrame({
        "byte_count":       byte_count,
        "packet_count":     pkt_count,
        "tcp_flags_any":    flags,
        "flow_duration_ms": dur_ms,
    }, index=df.index)


TIER_A_FN = {
    "lycos_ids2017": _tier_a_lycos,
    "cic_ids2018":   _tier_a_cic18,
    "unsw_nb15":     _tier_a_unsw,
    "ton_iot":       _tier_a_ton,
}

TIER_A_COLS = ["byte_count", "packet_count", "tcp_flags_any", "flow_duration_ms"]


def extract_tier_a(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Return a DataFrame with only the 4 Tier-A columns, aligned with df's index."""
    return TIER_A_FN[dataset](df)


def extract_tier_b(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Return Tier-B feature columns (already float32 in df after loading)."""
    return df[feature_cols].copy()
