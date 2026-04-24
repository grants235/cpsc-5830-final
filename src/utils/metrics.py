"""
Metrics: MCC (primary), macro-F1 (secondary), per-class F1, detection lag.
"""

import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, classification_report
from typing import List, Optional


CLASSES = ["Benign", "Reconnaissance", "DoS_DDoS", "Injection_Exploit",
           "BruteForce", "Botnet_C2"]


def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(matthews_corrcoef(y_true, y_pred))


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_per_class_f1(y_true_type: List[str], y_pred_binary: np.ndarray,
                          y_true_binary: np.ndarray) -> dict:
    """
    Per-attack-type F1 (each attack class vs benign).
    y_true_type: list of string label types for each edge.
    """
    results = {}
    y_true_type = np.array(y_true_type)
    for cls in CLASSES[1:]:  # skip Benign
        mask = (y_true_type == cls) | (y_true_type == "Benign")
        if mask.sum() == 0:
            results[cls] = float("nan")
            continue
        yt = y_true_binary[mask]
        yp = y_pred_binary[mask]
        results[cls] = float(f1_score(yt, yp, average="binary", zero_division=0))
    return results


def compute_detection_lag(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    src_ips: np.ndarray,
    timestamps: np.ndarray,
    min_campaign_len: int = 10,
) -> Optional[float]:
    """
    Median flows-to-first-correct-flag per attack campaign.
    Campaign = run of >= min_campaign_len consecutive attack flows from same src IP.
    Only meaningful for temporally-ordered predictions (TGN).
    """
    order = np.argsort(timestamps)
    yt = y_true_binary[order]
    yp = y_pred_binary[order]
    ips = src_ips[order]
    ts = timestamps[order]

    lags = []
    # Find campaigns per IP
    unique_ips = np.unique(ips[yt == 1])
    for ip in unique_ips:
        ip_mask = ips == ip
        ip_yt = yt[ip_mask]
        ip_yp = yp[ip_mask]

        # Find runs of consecutive attacks
        runs = []
        run_start = None
        for i, label in enumerate(ip_yt):
            if label == 1 and run_start is None:
                run_start = i
            elif label == 0 and run_start is not None:
                runs.append((run_start, i))
                run_start = None
        if run_start is not None:
            runs.append((run_start, len(ip_yt)))

        for start, end in runs:
            if (end - start) < min_campaign_len:
                continue
            # Find first correct prediction in campaign
            for j in range(start, end):
                if ip_yp[j] == 1:
                    lags.append(j - start)
                    break
            else:
                lags.append(end - start)  # never detected

    if not lags:
        return None
    return float(np.median(lags))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_type: Optional[List[str]] = None,
    src_ips: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
) -> dict:
    results = {
        "mcc":      compute_mcc(y_true, y_pred),
        "macro_f1": compute_macro_f1(y_true, y_pred),
    }
    if y_true_type is not None:
        results["per_class_f1"] = compute_per_class_f1(y_true_type, y_pred, y_true)
    if src_ips is not None and timestamps is not None:
        results["detection_lag"] = compute_detection_lag(y_true, y_pred, src_ips, timestamps)
    return results
