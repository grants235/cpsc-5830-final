"""
Per-dataset loaders.  Each returns a canonical DataFrame with columns:
  src_ip (str), dst_ip (str), timestamp (int64 µs),
  <feature cols> (float32), label_binary (uint8), label_type (str)

Timestamps are monotonically non-decreasing µs since Unix epoch.
Rows whose native label has no mapping in the 6-class taxonomy are dropped.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.data.label_map import LABEL_MAPS

log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2] / "data"

# ---------------------------------------------------------------------------
# Tier-A feature mapping per dataset (byte_count, packet_count, tcp_flags_any,
# flow_duration_ms) — used to build the 4-dim shared feature set.
# ---------------------------------------------------------------------------

UNSW_COLS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state",
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
    "service", "Sload", "Dload", "Spkts", "Dpkts",
    "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "Sjit", "Djit",
    "Stime", "Ltime", "Sintpkt", "Dintpkt",
    "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "label",
]

# Tier-B feature columns per dataset (all numeric, non-identifier).
LYCOS_FEATURE_COLS: List[str] = [
    "flow_duration", "down_up_ratio", "pkt_len_max", "pkt_len_min",
    "pkt_len_mean", "pkt_len_var", "bytes_per_s", "pkt_per_s",
    "fwd_pkt_per_s", "bwd_pkt_per_s", "fwd_pkt_cnt", "fwd_pkt_len_tot",
    "fwd_pkt_len_max", "fwd_pkt_len_min", "fwd_pkt_len_mean",
    "fwd_pkt_len_std", "fwd_pkt_hdr_len_tot", "fwd_pkt_hdr_len_min",
    "fwd_non_empty_pkt_cnt", "bwd_pkt_cnt", "bwd_pkt_len_tot",
    "bwd_pkt_len_max", "bwd_pkt_len_min", "bwd_pkt_len_mean",
    "bwd_pkt_len_std", "bwd_pkt_hdr_len_tot", "bwd_pkt_hdr_len_min",
    "bwd_non_empty_pkt_cnt", "iat_max", "iat_min", "iat_mean", "iat_std",
    "fwd_iat_tot", "fwd_iat_max", "fwd_iat_min", "fwd_iat_mean",
    "fwd_iat_std", "bwd_iat_tot", "bwd_iat_max", "bwd_iat_min",
    "bwd_iat_mean", "bwd_iat_std", "active_max", "active_min",
    "active_mean", "active_std", "idle_max", "idle_min", "idle_mean",
    "idle_std", "flag_SYN", "flag_fin", "flag_rst", "flag_ack",
    "flag_psh", "fwd_flag_psh", "bwd_flag_psh", "flag_urg",
    "fwd_flag_urg", "bwd_flag_urg", "flag_cwr", "flag_ece",
    "fwd_bulk_bytes_mean", "fwd_bulk_pkt_mean", "fwd_bulk_rate_mean",
    "bwd_bulk_bytes_mean", "bwd_bulk_pkt_mean", "bwd_bulk_rate_mean",
    "fwd_subflow_bytes_mean", "fwd_subflow_pkt_mean",
    "bwd_subflow_bytes_mean", "bwd_subflow_pkt_mean",
    "fwd_tcp_init_win_bytes", "bwd_tcp_init_win_bytes",
]

CIC18_FEATURE_COLS: List[str] = [
    "L7_PROTO", "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS",
]

UNSW_FEATURE_COLS: List[str] = [
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "Sjit", "Djit", "Sintpkt", "Dintpkt",
    "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
]

TON_FEATURE_COLS: List[str] = [
    "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
]

FEATURE_COLS = {
    "lycos_ids2017": LYCOS_FEATURE_COLS,
    "cic_ids2018":   CIC18_FEATURE_COLS,
    "unsw_nb15":     UNSW_FEATURE_COLS,
    "ton_iot":       TON_FEATURE_COLS,
}


def _apply_label_map(df: pd.DataFrame, raw_col: str, dataset: str) -> pd.DataFrame:
    lmap = LABEL_MAPS[dataset]
    before = len(df)
    df["label_type"] = df[raw_col].map(lmap)
    dropped = df["label_type"].isna().sum()
    if dropped > 0:
        pct = dropped / before * 100
        log.warning(f"[{dataset}] dropped {dropped} rows ({pct:.2f}%) with unmapped labels")
    df = df.dropna(subset=["label_type"]).copy()
    df["label_binary"] = (df["label_type"] != "Benign").astype("uint8")
    return df


def _ensure_monotonic_ts(ts: pd.Series) -> pd.Series:
    """Add row-index jitter so timestamps are non-decreasing."""
    arr = ts.values.copy()
    jitter = np.arange(len(arr), dtype=np.int64)
    arr = arr + jitter
    # enforce non-decreasing
    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            arr[i] = arr[i - 1] + 1
    return pd.Series(arr, index=ts.index, dtype=np.int64)


def load_lycos_ids2017(raw_dir: Path | None = None) -> Tuple[pd.DataFrame, List[str]]:
    path = (raw_dir or BASE / "lycos-ids2017") / "LycoS-IDS2017.csv"
    log.info(f"Loading LycoS-IDS2017 from {path}")
    df = pd.read_csv(path, low_memory=False)

    df = df.rename(columns={"src_addr": "src_ip", "dst_addr": "dst_ip"})
    # Sort by original µs timestamp so temporal split is chronological
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = _ensure_monotonic_ts(df["timestamp"].astype(np.int64))

    df = _apply_label_map(df, "label", "lycos_ids2017")

    feat_cols = [c for c in LYCOS_FEATURE_COLS if c in df.columns]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")

    return df[["src_ip", "dst_ip", "timestamp"] + feat_cols + ["label_binary", "label_type"]], feat_cols


def load_cic_ids2018(raw_dir: Path | None = None) -> Tuple[pd.DataFrame, List[str]]:
    path = (raw_dir or BASE / "cic-ids2018") / "NF-CSE-CIC-IDS2018.csv"
    log.info(f"Loading CSE-CIC-IDS2018 from {path}")
    df = pd.read_csv(path, low_memory=False)

    df = df.rename(columns={
        "IPV4_SRC_ADDR": "src_ip",
        "IPV4_DST_ADDR": "dst_ip",
    })

    # No timestamp column — synthesize from row index (1 ms per row, epoch base 2018-02-07)
    epoch_base = int(1518000000 * 1e6)  # 2018-02-07 in µs
    df["timestamp"] = epoch_base + np.arange(len(df), dtype=np.int64) * 1000

    df = _apply_label_map(df, "Attack", "cic_ids2018")

    feat_cols = [c for c in CIC18_FEATURE_COLS if c in df.columns]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")

    return df[["src_ip", "dst_ip", "timestamp"] + feat_cols + ["label_binary", "label_type"]], feat_cols


def load_unsw_nb15(raw_dir: Path | None = None) -> Tuple[pd.DataFrame, List[str]]:
    raw_dir = raw_dir or BASE / "unsw-nb15"
    files = sorted(raw_dir.glob("UNSW-NB15_[1-4].csv"))
    log.info(f"Loading UNSW-NB15 from {[str(f) for f in files]}")

    chunks = []
    for f in files:
        chunk = pd.read_csv(f, header=None, names=UNSW_COLS, low_memory=False)
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    df = df.rename(columns={"srcip": "src_ip", "dstip": "dst_ip"})

    # Stime is epoch seconds — sort chronologically before building graph
    df["Stime"] = pd.to_numeric(df["Stime"], errors="coerce").fillna(0)
    df = df.sort_values("Stime").reset_index(drop=True)
    df["timestamp"] = _ensure_monotonic_ts((df["Stime"] * 1e6).astype(np.int64))

    # Normalize attack_cat: strip whitespace
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    # Binary 0 rows with empty attack_cat are Benign
    df.loc[(df["label"].astype(str) == "0") & (df["attack_cat"].isin(["", "nan"])), "attack_cat"] = "Normal"

    df = _apply_label_map(df, "attack_cat", "unsw_nb15")

    feat_cols = [c for c in UNSW_FEATURE_COLS if c in df.columns]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")

    return df[["src_ip", "dst_ip", "timestamp"] + feat_cols + ["label_binary", "label_type"]], feat_cols


def load_ton_iot(raw_dir: Path | None = None) -> Tuple[pd.DataFrame, List[str]]:
    path = (raw_dir or BASE / "ton-iot") / "NF-ToN-IoT-v3.csv"
    log.info(f"Loading ToN-IoT from {path}")
    df = pd.read_csv(path, low_memory=False)

    df = df.rename(columns={
        "IPV4_SRC_ADDR": "src_ip",
        "IPV4_DST_ADDR": "dst_ip",
    })
    # Sort by flow start time so temporal split is chronological
    df = df.sort_values("FLOW_START_MILLISECONDS").reset_index(drop=True)
    # FLOW_START_MILLISECONDS → µs
    df["timestamp"] = _ensure_monotonic_ts(
        (df["FLOW_START_MILLISECONDS"].astype(np.int64) * 1000)
    )

    df = _apply_label_map(df, "Attack", "ton_iot")

    feat_cols = [c for c in TON_FEATURE_COLS if c in df.columns]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")

    return df[["src_ip", "dst_ip", "timestamp"] + feat_cols + ["label_binary", "label_type"]], feat_cols


LOADERS = {
    "lycos_ids2017": load_lycos_ids2017,
    "cic_ids2018":   load_cic_ids2018,
    "unsw_nb15":     load_unsw_nb15,
    "ton_iot":       load_ton_iot,
}
