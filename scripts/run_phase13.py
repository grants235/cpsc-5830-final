#!/usr/bin/env python3
"""
Phase 13 (spex13.md): Calibration Deep-Dive + Remaining Ablations.

Part A — Calibration analysis on existing checkpoints:
  A.1  Six new calibration methods applied to all checkpoints
  A.2  Per-fold winner analysis
  A.4  Output results/calibration_v2.csv

Part B — Missing ablation models:
  E13.1  TS-GIB raw features β=0.01  (all 4 folds)
  E13.2  TS-SAGE raw features         (all 4 folds)
  E13.3  TS-SAGE no-features ton_iot  (fills E8.1 gap)
  E13.4  GIB no-features β=0.01 LODO (all 4 folds)

Usage:
    python scripts/run_phase13.py --exp a1 [--target-exps E12.1 E8.1]
    python scripts/run_phase13.py --exp a2
    python scripts/run_phase13.py --exp b1_ts_gib_raw   [--beta 0.01]
    python scripts/run_phase13.py --exp b1_ts_sage_raw
    python scripts/run_phase13.py --exp b1_ts_sage_ton
    python scripts/run_phase13.py --exp b1_gib_nofeat   [--beta 0.01]
    python scripts/run_phase13.py --exp all_b1
    python scripts/run_phase13.py --exp all

    Add --no-dev to use full datasets.
    Add --folds lycos_ids2017 cic_ids2018  to restrict folds.
"""

import argparse
import copy
import csv
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split as _tts
from torch_geometric.data import Batch

from run_phase4 import ALL_FOLDS, _get_domain_labels
from run_phase10 import train_gib, _load_fold as _load_fold_real, _make_val_split as _make_val_split10
from run_phase12 import (
    _load_fold_struct, _graph_arrays, _delta_us,
    _ts_gib_forward_batch, _ts_gib_embed_batch,
    DELTA_SECS, BETA_SWEEP as BETA_SWEEP_12, DEFAULT_BETA, BETA_WARMUP,
    MAX_EPOCHS, PATIENCE, BATCH_SIZE, MAX_TRAIN_EDGES,
    MAX_VAL_EDGES, MAX_EVAL_EDGES, MAX_SUB_EDGES, NODE_FEAT_DIM, N_JOBS,
    _all_calibration_mccs, _calibrated_mcc, _oracle_mcc,
    _topk_threshold, _otsu_threshold,
)

from src.utils.logging import setup_logging, log_result, already_done, save_model, MODELS_DIR
from src.utils.seeding import seed_everything
from src.utils.metrics import compute_all_metrics, compute_mcc
from src.data.graph_builder import load_graph, combine_graphs
from src.data.temporal_subgraph import batch_build_subgraphs
from src.models.temporal_gnn import TS_GIB, TemporalEdgeSAGE
from src.models.gib_egraphsage import GIB_EGraphSAGE
from src.train.train_loops import _class_weights
from src.train.eval import eval_egraphsage

log = logging.getLogger(__name__)

FIGURES_DIR     = Path("results/figures/phase13")
INFERENCE_DIR   = Path("results/inference")
CAL_V2_CSV      = Path("results/calibration_v2.csv")
DEFAULT_BETA_13 = 0.01

# Target experiments for Part A (can be overridden via --target-exps)
DEFAULT_A1_TARGETS = [
    "E12.1_ts_gib_b0.001",
    "E12.1_ts_gib_b0.01",
    "E12.1_ts_gib_b0.1",
    "E12.2_ts_gib_anomaly_b0.01",
    "E8.1_ts_sage_d60",
    "E1.E_struct_only",
    "E13.1_ts_gib_raw_b0.01",
    "E13.2_ts_sage_raw",
    "E13.4_gib_nofeat_b0.01",
]


# ─────────────────────────────────────────────────────────────────────────────
# Part A helpers: val-score computation
# ─────────────────────────────────────────────────────────────────────────────

def _reproduce_val_split_p12(labels_np: np.ndarray, seed: int,
                              max_val: int = MAX_VAL_EDGES):
    """Reproduce the exact val split used in run_phase12._train_ts_gib."""
    _, vi = _tts(np.arange(len(labels_np)), test_size=0.2, random_state=seed,
                 stratify=labels_np)
    vi = np.array(vi, dtype=np.int64)
    if len(vi) > max_val:
        vi_lbl = labels_np[vi]
        _, vi_sub = _tts(vi, test_size=max_val / len(vi),
                         random_state=seed + 999, stratify=vi_lbl)
        vi = np.sort(np.array(vi_sub, dtype=np.int64))
    return vi


def _reproduce_val_split_p8(labels_np: np.ndarray, seed: int,
                             max_train_edges: int = MAX_TRAIN_EDGES):
    """Reproduce the exact val split used in run_phase8._train_temporal."""
    _, vi = _tts(np.arange(len(labels_np)), test_size=0.2, random_state=seed,
                 stratify=labels_np)
    vi = np.array(vi, dtype=np.int64)
    max_val = max_train_edges // 4
    if len(vi) > max_val:
        rng = np.random.RandomState(seed + 999)
        vi = rng.choice(vi, max_val, replace=False)
    return np.sort(vi)


@torch.no_grad()
def _val_scores_temporal(ckpt_path: Path, fold: dict, seed: int,
                          beta: float, dev: bool, device: str,
                          anomaly_scores_train=None) -> tuple:
    """
    Re-run val inference for a TS_GIB checkpoint.
    Returns (val_scores [V], val_labels [V]) or (None, None) on failure.
    """
    if not ckpt_path.exists():
        return None, None

    combined, _, _, _ = _load_fold_struct(fold, dev)
    labels_np = combined.edge_label.numpy()
    vi = _reproduce_val_split_p12(labels_np, seed)

    # Build model from state dict — detect TS_GIB vs TemporalEdgeSAGE by keys.
    state    = torch.load(ckpt_path, weights_only=True)
    is_ts_gib = "ctx_enc.0.weight" in state

    if is_ts_gib:
        has_q_enc = any(k.startswith("q_enc.") for k in state)
        q_edge_in = state["q_enc.0.weight"].shape[1] if has_q_enc else 1
        use_bn    = "to_dist.weight" in state
        dom_w     = state.get("domain_head.2.weight")
        num_doms  = dom_w.shape[0] if dom_w is not None else 0
        model     = TS_GIB(node_in=NODE_FEAT_DIM, ctx_edge_in=1, q_edge_in=q_edge_in,
                           hidden=128, use_bottleneck=use_bn, num_domains=num_doms)
    else:  # TemporalEdgeSAGE (E8.x / E9.x)
        edge_in   = state["edge_enc.0.weight"].shape[1]
        hidden    = state["edge_enc.0.weight"].shape[0]
        node_in   = state["conv1.lin_l.weight"].shape[1] - hidden
        q_edge_in = edge_in
        model     = TemporalEdgeSAGE(node_in=node_in, edge_in=edge_in, hidden=hidden)

    model.load_state_dict(state)
    model.eval().to(device)

    # Determine how to build q_ea for each batch:
    #   q_edge_in=1  → constant 1.0  (E12.1, E12.3, E8.x, E9.x)
    #   q_edge_in=2  → [1.0, anomaly_score]  (E12.2, E12.4)
    #   q_edge_in>2  → real quantile features  (E13.1 and similar)
    feat_q_np = None   # raw query features array, used when q_edge_in > 2
    if q_edge_in == 2 and anomaly_scores_train is None:
        anomaly_scores_train = np.zeros(len(labels_np), dtype=np.float32)
    elif q_edge_in > 2:
        combined_raw, _, _, _, _ = _load_fold_raw(fold, dev)
        feat_q_np = combined_raw.edge_attr_q.numpy().astype(np.float32)
        d = feat_q_np.shape[1]
        if d < q_edge_in:
            feat_q_np = np.concatenate(
                [feat_q_np, np.zeros((len(feat_q_np), q_edge_in - d), dtype=np.float32)], axis=1)
        elif d > q_edge_in:
            feat_q_np = feat_q_np[:, :q_edge_in]

    src_np, dst_np, time_np = _graph_arrays(combined)
    du = _delta_us(DELTA_SECS)

    all_scores, all_preds = [], []
    for start in range(0, len(vi), BATCH_SIZE):
        ids   = vi[start:start + BATCH_SIZE]
        if feat_q_np is not None:
            q_ea = _make_q_ea_raw(ids, feat_q_np, device)
        else:
            q_ea = _make_q_ea_val(ids, device, anomaly_scores_train)
        data_list = batch_build_subgraphs(
            src_np, dst_np, time_np,
            src_np[ids], dst_np[ids], time_np[ids],
            delta_us=du, max_edges=MAX_SUB_EDGES,
            node_feat_dim=NODE_FEAT_DIM, seed=seed, n_jobs=N_JOBS,
        )
        out = _ts_gib_forward_batch(model, data_list, q_ea, device)
        logits = out[0] if isinstance(out, tuple) else out
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_scores.append(probs)

    return np.concatenate(all_scores).astype(np.float32), labels_np[vi]


@torch.no_grad()
def _val_scores_standard(ckpt_path: Path, fold: dict, seed: int,
                          exp_id: str, dev: bool, device: str) -> tuple:
    """
    Re-run val inference for a non-temporal (EdgeAwareSAGE or GIB) checkpoint.
    Returns (val_scores [V], val_labels [V]) or (None, None) on failure.
    """
    if not ckpt_path.exists():
        return None, None

    # Detect model type from state dict
    state = torch.load(ckpt_path, weights_only=True)
    is_gib = any("to_dist" in k for k in state)

    use_quantile = not any(x in exp_id for x in ("E1.E", "struct", "nofeat",
                                                   "_no_feat", "ts_sage", "ts_gib"))
    combined, test_graph = _load_fold_real(fold, dev, use_quantile=use_quantile)
    val_split = _make_val_split10(combined, seed)
    vi = np.array(val_split["val"], dtype=np.int64)

    x  = combined.x.to(device)
    ei = combined.edge_index.to(device)
    ea = (combined.edge_attr_q if use_quantile else combined.edge_attr).to(device)

    # Read dims from checkpoint — avoids mismatch when graph features are wider
    # than what the model was trained with (e.g. nofeat models have edge_in=1).
    ck_key = "encoder.edge_enc.0.weight" if "encoder.edge_enc.0.weight" in state else "edge_enc.0.weight"
    ck_edge_in = state[ck_key].shape[1]
    ck_node_in = combined.x.shape[1]   # node features are always structural, no mismatch

    if ea.shape[1] != ck_edge_in:
        # Pad or crop graph edge features to match checkpoint's expected edge_in
        if ea.shape[1] < ck_edge_in:
            pad = torch.zeros(ea.shape[0], ck_edge_in - ea.shape[1], device=ea.device)
            ea  = torch.cat([ea, pad], dim=1)
        else:
            ea = ea[:, :ck_edge_in]

    if is_gib:
        model = GIB_EGraphSAGE(node_in=ck_node_in, edge_in=ck_edge_in)
    else:
        from src.models.egraphsage import EdgeAwareSAGE
        model = EdgeAwareSAGE(node_in=ck_node_in, edge_in=ck_edge_in)
    model.load_state_dict(state)
    model.eval().to(device)

    all_scores = []
    for s in range(0, len(vi), 50_000):
        idx = vi[s:s + 50_000]
        logits = model(x, ei[:, idx], ea[idx])
        probs  = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        all_scores.append(probs)

    labels_np = combined.edge_label.numpy()
    return np.concatenate(all_scores).astype(np.float32), labels_np[vi]


@torch.no_grad()
def _test_scores_temporal(ckpt_path: Path, fold: dict, dev: bool, device: str) -> tuple:
    """
    Re-run test inference for a temporal checkpoint (TS_GIB or TemporalEdgeSAGE).
    Mirrors _val_scores_temporal but runs on the test graph.
    Returns (test_scores [N], test_labels [N]) or (None, None) on failure.
    """
    if not ckpt_path.exists():
        return None, None

    state     = torch.load(ckpt_path, weights_only=True)
    is_ts_gib = "ctx_enc.0.weight" in state

    if is_ts_gib:
        has_q_enc = any(k.startswith("q_enc.") for k in state)
        q_edge_in = state["q_enc.0.weight"].shape[1] if has_q_enc else 1
        use_bn    = "to_dist.weight" in state
        dom_w     = state.get("domain_head.2.weight")
        num_doms  = dom_w.shape[0] if dom_w is not None else 0
        model     = TS_GIB(node_in=NODE_FEAT_DIM, ctx_edge_in=1, q_edge_in=q_edge_in,
                           hidden=128, use_bottleneck=use_bn, num_domains=num_doms)
    else:
        edge_in   = state["edge_enc.0.weight"].shape[1]
        hidden    = state["edge_enc.0.weight"].shape[0]
        node_in   = state["conv1.lin_l.weight"].shape[1] - hidden
        q_edge_in = edge_in
        model     = TemporalEdgeSAGE(node_in=node_in, edge_in=edge_in, hidden=hidden)

    model.load_state_dict(state)
    model.eval().to(device)

    # Load test graph; real quantile features needed only for q_edge_in > 2 (E13.1)
    feat_q_np = None
    anom_np   = None
    if is_ts_gib and q_edge_in > 2:
        _, test_graph, _, _, _ = _load_fold_raw(fold, dev)
        feat_q_np = test_graph.edge_attr_q.numpy().astype(np.float32)
        d = feat_q_np.shape[1]
        if d < q_edge_in:
            feat_q_np = np.concatenate(
                [feat_q_np, np.zeros((len(feat_q_np), q_edge_in - d), dtype=np.float32)], axis=1)
        elif d > q_edge_in:
            feat_q_np = feat_q_np[:, :q_edge_in]
    else:
        _, test_graph, _, _ = _load_fold_struct(fold, dev)
        if is_ts_gib and q_edge_in == 2:
            anom_np = np.zeros(test_graph.edge_label.shape[0], dtype=np.float32)

    src_np, dst_np, time_np = _graph_arrays(test_graph)
    labels_np = test_graph.edge_label.numpy()
    n_total   = len(labels_np)
    du        = _delta_us(DELTA_SECS)

    if n_total > MAX_EVAL_EDGES:
        _, eval_idx = _tts(np.arange(n_total),
                           test_size=MAX_EVAL_EDGES / n_total,
                           random_state=42, stratify=labels_np)
        eval_idx = np.sort(np.array(eval_idx, dtype=np.int64))
        log.info(f"  Test capped: {n_total:,} → {len(eval_idx):,}")
    else:
        eval_idx = np.arange(n_total, dtype=np.int64)

    log.info(f"  Extracting {len(eval_idx):,} test subgraphs (n_jobs={N_JOBS}) …")
    t0       = time.time()
    all_data = batch_build_subgraphs(
        src_np, dst_np, time_np,
        src_np[eval_idx], dst_np[eval_idx], time_np[eval_idx],
        delta_us=du, max_edges=MAX_SUB_EDGES,
        node_feat_dim=NODE_FEAT_DIM, seed=0, n_jobs=N_JOBS,
    )
    log.info(f"  Subgraph extraction: {time.time()-t0:.1f}s")

    all_scores = []
    for start in range(0, len(eval_idx), BATCH_SIZE):
        ids_b = eval_idx[start:start + BATCH_SIZE]
        if feat_q_np is not None:
            q_ea = _make_q_ea_raw(ids_b, feat_q_np, device)
        else:
            q_ea = _make_q_ea_val(ids_b, device, anom_np)
        dl     = all_data[start:start + BATCH_SIZE]
        out    = _ts_gib_forward_batch(model, dl, q_ea, device)
        logits = out[0] if isinstance(out, tuple) else out
        all_scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())

    return np.concatenate(all_scores).astype(np.float32), labels_np[eval_idx]


def _make_q_ea_val(ids: np.ndarray, device: str,
                    anomaly_scores=None) -> torch.Tensor:
    """Query-edge features for val inference (mirrors run_phase12._make_q_ea)."""
    B = len(ids)
    if anomaly_scores is None:
        return torch.ones(B, 1, device=device)
    a = torch.as_tensor(anomaly_scores[ids], dtype=torch.float32, device=device)
    return torch.stack([torch.ones(B, device=device), a], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Part A: six new calibration methods
# ─────────────────────────────────────────────────────────────────────────────

def _val_anchored(val_scores, val_labels, test_scores, test_labels, n_thresh=300):
    """Method 1 — threshold T_val that maximises val MCC, applied to test."""
    thresholds = np.linspace(val_scores.min(), val_scores.max(), n_thresh)
    best_t, best_m = float(np.median(val_scores)), -2.0
    for t in thresholds:
        m = compute_mcc(val_labels, (val_scores >= t).astype(int))
        if m > best_m:
            best_m, best_t = m, float(t)
    return compute_mcc(test_labels, (test_scores >= best_t).astype(int)), best_t


def _znorm_val_anchored(val_scores, val_labels, test_scores, test_labels, n_thresh=300):
    """Method 2 — z-normalise test with val mean/std, then apply val-anchored T."""
    mu  = float(val_scores.mean())
    std = float(val_scores.std()) + 1e-8
    z_val  = (val_scores - mu) / std
    z_test = (test_scores - mu) / std
    thresholds = np.linspace(z_val.min(), z_val.max(), n_thresh)
    best_t, best_m = 0.0, -2.0
    for t in thresholds:
        m = compute_mcc(val_labels, (z_val >= t).astype(int))
        if m > best_m:
            best_m, best_t = m, float(t)
    return compute_mcc(test_labels, (z_test >= best_t).astype(int)), best_t


def _platt_scaling(val_scores, val_labels, test_scores, test_labels):
    """Method 3 — Temperature scaling (Platt). Fit (T, b) on val NLL."""
    eps = 1e-7
    v_logit = np.log(np.clip(val_scores, eps, 1-eps) /
                     (1 - np.clip(val_scores, eps, 1-eps)))
    t_logit = np.log(np.clip(test_scores, eps, 1-eps) /
                     (1 - np.clip(test_scores, eps, 1-eps)))
    vl = val_labels.astype(np.float64)

    def nll(params):
        T, b = params[0], params[1]
        p = 1.0 / (1.0 + np.exp(-(v_logit - b) / (abs(T) + 1e-8)))
        p = np.clip(p, eps, 1-eps)
        return -np.mean(vl * np.log(p) + (1 - vl) * np.log(1 - p))

    try:
        res = minimize(nll, x0=[1.0, 0.0], method="L-BFGS-B",
                       bounds=[(0.01, 50.0), (-10.0, 10.0)], options={"maxiter": 200})
        T, b = float(res.x[0]), float(res.x[1])
    except Exception:
        T, b = 1.0, 0.0

    p_cal = 1.0 / (1.0 + np.exp(-(t_logit - b) / (abs(T) + 1e-8)))
    return compute_mcc(test_labels, (p_cal >= 0.5).astype(int)), T, b


def _bbse(val_scores, val_labels, test_scores, test_labels):
    """Method 4 — BBSE prevalence estimation (Lipton et al. 2018)."""
    val_preds = (val_scores >= 0.5).astype(int)
    P  = float(val_labels.sum())
    N_ = float(len(val_labels)) - P
    if P == 0 or N_ == 0:
        return compute_mcc(test_labels, (test_scores >= 0.5).astype(int)), 0.0

    TP  = float(((val_preds == 1) & (val_labels == 1)).sum())
    FP  = float(((val_preds == 1) & (val_labels == 0)).sum())
    TPR = TP / (P + 1e-8)
    FPR = FP / (N_ + 1e-8)
    q_test = float((test_scores >= 0.5).mean())

    denom = TPR - FPR
    if abs(denom) < 0.05:
        pi_hat = float(val_labels.mean())
    else:
        pi_hat = float(np.clip((q_test - FPR) / denom, 0.01, 0.99))

    k = max(1, int(round(pi_hat * len(test_scores))))
    t = _topk_threshold(test_scores, k)
    return compute_mcc(test_labels, (test_scores >= t).astype(int)), pi_hat


def _gmm_logit(test_scores, test_labels):
    """Method 5 — 2-component GMM on logits (pre-sigmoid, cleaner bimodality)."""
    eps = 1e-7
    logits = np.log(np.clip(test_scores, eps, 1-eps) /
                    (1 - np.clip(test_scores, eps, 1-eps)))
    sub = logits
    if len(logits) > 50_000:
        sub = np.random.RandomState(42).choice(logits, 50_000, replace=False)
    try:
        gmm = GaussianMixture(n_components=2, n_init=1, random_state=42)
        gmm.fit(sub.reshape(-1, 1))
        means = sorted(gmm.means_.flatten())
        t_logit = float(np.mean(means))
        t_prob  = 1.0 / (1.0 + np.exp(-t_logit))
        return compute_mcc(test_labels, (test_scores >= t_prob).astype(int)), t_prob
    except Exception:
        return float("nan"), 0.5


def _ensemble_majority(val_scores, val_labels, test_scores, test_labels, p_src):
    """Method 6 — majority-vote ensemble of topk, val-anchor, otsu, BBSE."""
    # topk at source rate
    k1 = max(1, int(round(p_src * len(test_scores))))
    p1 = (test_scores >= _topk_threshold(test_scores, k1)).astype(int)

    # val-anchor
    thresholds = np.linspace(val_scores.min(), val_scores.max(), 200)
    best_t, best_m = float(np.median(val_scores)), -2.0
    for t in thresholds:
        m = compute_mcc(val_labels, (val_scores >= t).astype(int))
        if m > best_m:
            best_m, best_t = m, float(t)
    p2 = (test_scores >= best_t).astype(int)

    # otsu on test
    p3 = (test_scores >= _otsu_threshold(test_scores)).astype(int)

    # bbse
    P  = float(val_labels.sum())
    N_ = float(len(val_labels)) - P
    if P > 0 and N_ > 0:
        TPR = float((((val_scores >= 0.5).astype(int) == 1) & (val_labels == 1)).sum()) / (P + 1e-8)
        FPR = float((((val_scores >= 0.5).astype(int) == 1) & (val_labels == 0)).sum()) / (N_ + 1e-8)
        q_t = float((test_scores >= 0.5).mean())
        denom = TPR - FPR
        pi_hat = float(np.clip((q_t - FPR) / denom, 0.01, 0.99)) if abs(denom) >= 0.05 else p_src
    else:
        pi_hat = p_src
    k4 = max(1, int(round(pi_hat * len(test_scores))))
    p4 = (test_scores >= _topk_threshold(test_scores, k4)).astype(int)

    votes = p1 + p2 + p3 + p4
    return compute_mcc(test_labels, (votes >= 2).astype(int))


def _all_cal_methods_v2(val_scores, val_labels, test_scores, test_labels, p_src):
    """
    Compute all Phase 13 calibration methods.
    val_scores/val_labels may be None — methods that need them return nan.
    Returns dict: method_name → mcc.
    """
    results = {}

    # Existing methods (reuse from run_phase12)
    existing = _all_calibration_mccs(test_scores, test_labels, p_src)
    results.update({f"p11_{k}": v for k, v in existing.items()})

    if val_scores is None or val_labels is None:
        for name in ("val_anchor", "znorm_val", "platt", "bbse", "ensemble"):
            results[name] = float("nan")
    else:
        m1, _ = _val_anchored(val_scores, val_labels, test_scores, test_labels)
        results["val_anchor"] = m1

        m2, _ = _znorm_val_anchored(val_scores, val_labels, test_scores, test_labels)
        results["znorm_val"] = m2

        m3, _, _ = _platt_scaling(val_scores, val_labels, test_scores, test_labels)
        results["platt"] = m3

        m4, _ = _bbse(val_scores, val_labels, test_scores, test_labels)
        results["bbse"] = m4

        results["ensemble"] = _ensemble_majority(
            val_scores, val_labels, test_scores, test_labels, p_src)

    m5, _ = _gmm_logit(test_scores, test_labels)
    results["gmm_logit"] = m5

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Part A: calibration sweep
# ─────────────────────────────────────────────────────────────────────────────

def _is_temporal(exp_id: str) -> bool:
    return any(exp_id.startswith(p) for p in
               ("E8.", "E9.", "E12.", "E13.1", "E13.2", "E13.3"))


def _load_inference_file(path: Path) -> dict | None:
    """Load a Phase 11 inference file. Returns None if not found/corrupt."""
    if not path.exists():
        return None
    try:
        d = torch.load(path, weights_only=False)
        scores = np.asarray(d["scores"], dtype=np.float32)
        labels = np.asarray(d["labels"], dtype=np.int32)
        return {"scores": scores, "labels": labels,
                "method": d.get("method", ""), "seed": d.get("seed", 0),
                "test_fold": d.get("test_fold", "")}
    except Exception as e:
        log.warning(f"  Could not load {path}: {e}")
        return None


def _infer_file_for(exp_id: str, seed: int, test_fold: str) -> Path:
    """Guess the Phase 11 inference file path for a checkpoint."""
    stem = f"{exp_id}_seed{seed}_test{test_fold}"
    return INFERENCE_DIR / f"{stem}.pt"


def _get_p_src_from_csv(exp_id: str, seed: int, test_fold: str) -> float:
    """Look up logged p_src from results.csv; fall back to 0.2."""
    path = Path("results/results.csv")
    if not path.exists():
        return 0.2
    with open(path) as f:
        for row in csv.DictReader(f):
            if (row["experiment_id"] == exp_id and int(row["seed"]) == seed
                    and row["test_dataset"] == test_fold
                    and row["metric"] == "p_src"):
                return float(row["value"])
    return 0.2


def run_a1_calibration_sweep(target_exps: list = None, dev: bool = True):
    """
    For each target experiment × seed × fold:
      - Load test scores (from Phase 11 inference dir, or skip)
      - Compute val scores (re-run inference if needed)
      - Compute all calibration methods
      - Append to calibration_v2.csv
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_exps = target_exps or DEFAULT_A1_TARGETS
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    CAL_V2_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id", "seed", "test_fold",
        "p11_reported_0.5", "p11_topk_src_rate", "p11_otsu",
        "p11_gmm", "p11_topk_10pct", "p11_topk_20pct", "p11_topk_30pct",
        "p11_oracle",
        "val_anchor", "znorm_val", "platt", "bbse", "gmm_logit", "ensemble",
        "oracle_mcc", "auroc",
    ]

    # Read already-computed rows to skip
    done_keys = set()
    if CAL_V2_CSV.exists():
        with open(CAL_V2_CSV) as f:
            for row in csv.DictReader(f):
                done_keys.add((row["experiment_id"], row["seed"], row["test_fold"]))

    with open(CAL_V2_CSV, "a", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, extrasaction="ignore")
        if not done_keys:
            writer.writeheader()

        for exp_id in target_exps:
            log.info(f"\n=== A.1 calibration: {exp_id} ===")
            temporal = _is_temporal(exp_id)

            # Discover which (seed, fold) pairs we have checkpoints for
            ckpt_pattern = f"{exp_id}_seed*_test*.pt"
            ckpts = sorted(MODELS_DIR.glob(ckpt_pattern))
            if not ckpts:
                log.warning(f"  No checkpoints matching {ckpt_pattern}")
                continue

            for ckpt in ckpts:
                # Parse seed and fold from filename
                stem  = ckpt.stem
                parts = stem.split("_seed")
                if len(parts) < 2:
                    continue
                tail  = parts[-1].split("_test")
                if len(tail) < 2:
                    continue
                try:
                    seed      = int(tail[0])
                    test_fold = tail[1]
                except ValueError:
                    continue

                if (exp_id, str(seed), test_fold) in done_keys:
                    log.info(f"  Skip {exp_id} seed={seed} fold={test_fold} (done)")
                    continue

                fold = next((f for f in ALL_FOLDS if f["test"] == test_fold), None)
                if fold is None:
                    continue

                log.info(f"  Processing {exp_id} seed={seed} fold={test_fold}")

                # ── Load or compute test scores ─────────────────────────────
                inf_path = _infer_file_for(exp_id, seed, test_fold)
                inf_data = _load_inference_file(inf_path)

                if inf_data is None:
                    if temporal:
                        log.info(f"  No inference file; computing test scores on the fly …")
                        try:
                            test_scores, test_labels = _test_scores_temporal(
                                ckpt, fold, dev, device)
                            if test_scores is None:
                                continue
                            beta = float(exp_id.split("_b")[-1]) if "_b" in exp_id else DEFAULT_BETA_13
                            log.info(f"  Computing val scores (temporal, n_jobs={N_JOBS}) …")
                            val_scores, val_labels = _val_scores_temporal(
                                ckpt, fold, seed, beta, dev, device)
                        except Exception as e:
                            log.warning(f"  Failed: {e}")
                            continue
                    else:
                        log.info(f"  No inference file; running batch inference …")
                        try:
                            vs_standard, vl_standard = _val_scores_standard(
                                ckpt, fold, seed, exp_id, dev, device)
                            if vs_standard is None:
                                continue
                            use_q = not any(x in exp_id for x in
                                            ("E1.E", "struct", "nofeat"))
                            _, test_graph = _load_fold_real(fold, dev, use_quantile=use_q)
                            state = torch.load(ckpt, weights_only=True)
                            ck_key = ("encoder.edge_enc.0.weight"
                                      if "encoder.edge_enc.0.weight" in state
                                      else "edge_enc.0.weight")
                            ck_edge_in = state[ck_key].shape[1]
                            ea = (test_graph.edge_attr_q if use_q else test_graph.edge_attr)
                            if ea.shape[1] != ck_edge_in:
                                if ea.shape[1] < ck_edge_in:
                                    pad = torch.zeros(ea.shape[0], ck_edge_in - ea.shape[1])
                                    ea  = torch.cat([ea, pad], dim=1)
                                else:
                                    ea = ea[:, :ck_edge_in]
                            is_gib = any("to_dist" in k for k in state)
                            from src.models.egraphsage import EdgeAwareSAGE
                            if is_gib:
                                m = GIB_EGraphSAGE(node_in=test_graph.x.shape[1],
                                                   edge_in=ck_edge_in)
                            else:
                                m = EdgeAwareSAGE(node_in=test_graph.x.shape[1],
                                                  edge_in=ck_edge_in)
                            m.load_state_dict(state)
                            m.eval().to(device)
                            # Use pre-cropped ea directly — eval_egraphsage would
                            # re-read test_graph.edge_attr and ignore the crop.
                            x_t  = test_graph.x.to(device)
                            ei_t = test_graph.edge_index.to(device)
                            ea_t = ea.to(device)
                            ts_scores, ts_preds = [], []
                            with torch.no_grad():
                                for s in range(0, ei_t.shape[1], 50_000):
                                    lg = m(x_t, ei_t[:, s:s+50_000], ea_t[s:s+50_000])
                                    ts_scores.append(
                                        torch.softmax(lg, dim=-1)[:, 1].cpu().numpy())
                            test_scores = np.concatenate(ts_scores).astype(np.float32)
                            test_labels = test_graph.edge_label.numpy()
                            val_scores, val_labels = vs_standard, vl_standard
                        except Exception as e:
                            log.warning(f"  Failed: {e}")
                            continue
                else:
                    test_scores = inf_data["scores"]
                    test_labels = inf_data["labels"]

                    # ── Compute val scores ──────────────────────────────────
                    if temporal:
                        beta = float(exp_id.split("_b")[-1]) if "_b" in exp_id else DEFAULT_BETA_13
                        log.info(f"  Computing val scores (temporal, n_jobs={N_JOBS}) …")
                        val_scores, val_labels = _val_scores_temporal(
                            ckpt, fold, seed, beta, dev, device)
                    else:
                        log.info(f"  Computing val scores (standard) …")
                        val_scores, val_labels = _val_scores_standard(
                            ckpt, fold, seed, exp_id, dev, device)

                if test_scores is None or len(test_scores) == 0:
                    continue

                # ── Calibration methods ─────────────────────────────────────
                p_src = _get_p_src_from_csv(exp_id, seed, test_fold)
                cal   = _all_cal_methods_v2(
                    val_scores, val_labels, test_scores, test_labels, p_src)
                orc, _ = _oracle_mcc(test_scores, test_labels)
                try:
                    auroc = float(roc_auc_score(test_labels, test_scores))
                except Exception:
                    auroc = float("nan")

                row = {"experiment_id": exp_id, "seed": str(seed),
                       "test_fold": test_fold, "oracle_mcc": f"{orc:.6f}",
                       "auroc": f"{auroc:.6f}"}
                for k, v in cal.items():
                    row[k] = f"{v:.6f}" if not np.isnan(v) else "nan"
                writer.writerow(row)
                f_out.flush()
                done_keys.add((exp_id, str(seed), test_fold))
                log.info(f"  Done: oracle={orc:.4f}  auroc={auroc:.4f}"
                          f"  val_anchor={cal.get('val_anchor', float('nan')):.4f}"
                          f"  bbse={cal.get('bbse', float('nan')):.4f}")

    log.info(f"\n  calibration_v2.csv → {CAL_V2_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
# Part A.2: per-fold winner analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_a2_winner_analysis(exp_id_filter: str = "E12.1_ts_gib_b0.01"):
    """
    Read calibration_v2.csv and report:
      - Per-fold best calibration method and its MCC
      - Oracle-recovery % for each method × fold
      - Whether a single dominant method exists
    """
    if not CAL_V2_CSV.exists():
        log.error(f"  {CAL_V2_CSV} not found — run A.1 first.")
        return

    CAL_METHODS = [
        "p11_topk_src_rate", "p11_otsu", "p11_gmm",
        "p11_topk_10pct", "p11_topk_20pct", "p11_topk_30pct",
        "val_anchor", "znorm_val", "platt", "bbse", "gmm_logit", "ensemble",
    ]

    # Aggregate: for each (exp_id, fold) keep the best seed's row
    rows = {}
    with open(CAL_V2_CSV) as f:
        for row in csv.DictReader(f):
            if exp_id_filter and not row["experiment_id"].startswith(exp_id_filter.split("_b")[0]):
                continue
            key = (row["experiment_id"], row["test_fold"])
            try:
                orc = float(row.get("oracle_mcc", "nan"))
            except ValueError:
                orc = float("nan")
            # Keep the row with highest oracle MCC (best seed if multi-seed)
            if key not in rows or orc > float(rows[key].get("oracle_mcc", -999)):
                rows[key] = row

    if not rows:
        log.warning(f"  No rows matching '{exp_id_filter}' in {CAL_V2_CSV}")
        return

    log.info(f"\n{'='*70}")
    log.info(f"  A.2 Winner analysis — filter: '{exp_id_filter}'")
    log.info(f"{'='*70}")

    # Per-fold best method
    fold_winners: dict = {}
    for (eid, fold), row in sorted(rows.items()):
        try:
            orc = float(row.get("oracle_mcc", "nan"))
        except ValueError:
            orc = float("nan")

        method_mccs = {}
        for m in CAL_METHODS:
            try:
                v = float(row.get(m, "nan"))
                if not np.isnan(v):
                    method_mccs[m] = v
            except (ValueError, TypeError):
                pass

        if not method_mccs:
            continue

        best_m    = max(method_mccs, key=method_mccs.get)
        best_mcc  = method_mccs[best_m]
        rep_mcc   = float(row.get("p11_reported_0.5", 0))
        oracle_rec = ((best_mcc - rep_mcc) / (orc - rep_mcc)
                      if abs(orc - rep_mcc) > 0.01 else float("nan"))

        fold_winners.setdefault(fold, []).append((eid, best_m, best_mcc, orc, oracle_rec))
        log.info(f"  {fold:<22} {eid}")
        log.info(f"    oracle={orc:.4f}  reported={rep_mcc:.4f}")
        for mn in sorted(method_mccs, key=method_mccs.get, reverse=True)[:5]:
            rec = ((method_mccs[mn] - rep_mcc) / (orc - rep_mcc)
                   if abs(orc - rep_mcc) > 0.01 else float("nan"))
            log.info(f"    {mn:<22} mcc={method_mccs[mn]:.4f}  "
                      f"recovery={rec:.1%}" if not np.isnan(rec) else
                      f"    {mn:<22} mcc={method_mccs[mn]:.4f}")

    # Overall winner count
    method_wins: dict = {}
    for fold, winners in fold_winners.items():
        for _, best_m, _, _, _ in winners:
            method_wins[best_m] = method_wins.get(best_m, 0) + 1

    log.info(f"\n  Method win counts across folds:")
    for m, cnt in sorted(method_wins.items(), key=lambda x: -x[1]):
        log.info(f"    {m:<25} wins={cnt}")

    top_winner = max(method_wins, key=method_wins.get) if method_wins else "none"
    top_wins   = method_wins.get(top_winner, 0)
    total_folds = len(fold_winners)
    if total_folds > 0 and top_wins >= total_folds:
        log.info(f"\n  OUTCOME 1: '{top_winner}' dominates all {total_folds} folds.")
    elif total_folds > 0 and top_wins >= total_folds * 0.6:
        log.info(f"\n  OUTCOME 2: '{top_winner}' wins {top_wins}/{total_folds} folds. "
                  f"Use ensemble for robustness.")
    else:
        log.info(f"\n  OUTCOME 3: No dominant method. "
                  f"Calibration is fold-specific; report ensemble.")

    return top_winner


# ─────────────────────────────────────────────────────────────────────────────
# Part B helpers: raw-features data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_fold_raw(fold: dict, dev: bool):
    """
    Load fold with real quantile-encoded features.
    Returns (combined, test_graph, feat_dim) where feat_dim is the
    common feature dim after alignment.
    """
    train_graphs = [load_graph(ds, tier="B", dev=dev) for ds in fold["train"]]
    combined     = combine_graphs(train_graphs)

    test_graph = load_graph(fold["test"], tier="B", dev=dev)
    feat_dim   = combined.edge_attr_q.shape[1]
    d = test_graph.edge_attr_q.shape[1]
    if d < feat_dim:
        test_graph.edge_attr_q = torch.cat(
            [test_graph.edge_attr_q,
             torch.zeros(test_graph.edge_attr_q.shape[0], feat_dim - d)], dim=1)
    elif d > feat_dim:
        test_graph.edge_attr_q = test_graph.edge_attr_q[:, :feat_dim]

    return combined, test_graph, fold["train"], fold["test"], feat_dim


def _make_q_ea_raw(ids: np.ndarray, feat_q: np.ndarray, device: str) -> torch.Tensor:
    """Query-edge features from raw quantile features [E, feat_dim]."""
    return torch.as_tensor(feat_q[ids], dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Part B training loop (raw features)
# ─────────────────────────────────────────────────────────────────────────────

def _train_ts_raw(
    model: TS_GIB,
    graph,
    feat_q_np: np.ndarray,
    device: str,
    seed: int,
    exp_id: str,
    test_dset: str,
    delta_us: int,
    beta_max: float = 0.0,
    domain_labels=None,
    max_edges: int = MAX_SUB_EDGES,
    epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    max_train_edges: int = MAX_TRAIN_EDGES,
    max_val_edges: int = MAX_VAL_EDGES,
    warmup_epochs: int = BETA_WARMUP,
    n_jobs: int = N_JOBS,
) -> tuple:
    """
    Train TS_GIB with real edge features as query-edge features.
    feat_q_np: [E_combined, feat_dim] quantile features for every training edge.
    Returns (best_state_dict, p_src).
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    cw        = _class_weights(graph.edge_label, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    dom_crit  = nn.CrossEntropyLoss() if domain_labels is not None else None

    src_np, dst_np, time_np = _graph_arrays(graph)
    labels_np = graph.edge_label.numpy()
    n         = len(labels_np)

    _, vi = _tts(np.arange(n), test_size=0.2, random_state=seed, stratify=labels_np)
    ti_full = np.setdiff1d(np.arange(n), vi)
    vi = np.array(vi, dtype=np.int64)

    if len(vi) > max_val_edges:
        _, vi_sub = _tts(vi, test_size=max_val_edges / len(vi),
                         random_state=seed + 999, stratify=labels_np[vi])
        vi = np.sort(np.array(vi_sub, dtype=np.int64))

    p_src = float(labels_np[vi].mean())
    log.info(f"  train={len(ti_full):,}  val={len(vi):,}  p_src={p_src:.4f}")

    # Pre-cache val subgraphs
    log.info(f"  Pre-extracting {len(vi):,} val subgraphs …")
    t_c = time.time()
    val_cache = batch_build_subgraphs(
        src_np, dst_np, time_np,
        src_np[vi], dst_np[vi], time_np[vi],
        delta_us=delta_us, max_edges=max_edges,
        node_feat_dim=NODE_FEAT_DIM, seed=seed, n_jobs=n_jobs,
    )
    log.info(f"  Val cache ready in {time.time()-t_c:.1f}s")

    best_mcc, best_state, pat_cnt = -2.0, None, 0
    ep_rng = np.random.RandomState(seed)
    use_domain = domain_labels is not None and model.domain_head is not None

    for epoch in range(epochs):
        beta = beta_max * min(1.0, (epoch + 1) / max(1, warmup_epochs))
        model.train()

        if len(ti_full) > max_train_edges:
            _, ti_arr = _tts(ti_full, test_size=max_train_edges / len(ti_full),
                             random_state=ep_rng.randint(0, 2**31),
                             stratify=labels_np[ti_full])
            ti_arr = np.array(ti_arr, dtype=np.int64)
        else:
            ti_arr = ti_full.copy()
        ep_rng.shuffle(ti_arr)

        ep_loss, n_batches = 0.0, 0
        for start in range(0, len(ti_arr), batch_size):
            ids   = ti_arr[start:start + batch_size]
            yl    = torch.as_tensor(labels_np[ids], dtype=torch.long, device=device)
            q_ea  = _make_q_ea_raw(ids, feat_q_np, device)
            data_list = batch_build_subgraphs(
                src_np, dst_np, time_np,
                src_np[ids], dst_np[ids], time_np[ids],
                delta_us=delta_us, max_edges=max_edges,
                node_feat_dim=NODE_FEAT_DIM, seed=seed, n_jobs=n_jobs,
            )
            if use_domain:
                dom_yl = domain_labels[ids].to(device)
                atk, dom, kl = _ts_gib_forward_batch(
                    model, data_list, q_ea, device, use_domain=True)
                loss = criterion(atk, yl) + dom_crit(dom, dom_yl) + beta * kl
            else:
                logits, kl = _ts_gib_forward_batch(
                    model, data_list, q_ea, device, use_domain=False)
                loss = criterion(logits, yl) + beta * kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item(); n_batches += 1

        # Validation
        model.eval()
        all_preds = []
        for start in range(0, len(vi), batch_size):
            ids  = vi[start:start + batch_size]
            q_ea = _make_q_ea_raw(ids, feat_q_np, device)
            dl   = val_cache[start:start + batch_size]
            with torch.no_grad():
                logits, _ = _ts_gib_forward_batch(model, dl, q_ea, device)
            all_preds.append(logits.argmax(1).cpu().numpy())

        val_mcc = compute_mcc(labels_np[vi], np.concatenate(all_preds))
        log.info(f"  epoch {epoch+1:02d}  β={beta:.4f}"
                  f"  loss={ep_loss/max(1,n_batches):.4f}  val_mcc={val_mcc:.4f}")

        if val_mcc > best_mcc:
            best_mcc   = val_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt    = 0
        else:
            if epoch >= 3:
                pat_cnt += 1
                if pat_cnt >= patience:
                    log.info(f"  Early stop epoch {epoch+1}")
                    break

    log.info(f"  Best val MCC: {best_mcc:.4f}  p_src={p_src:.4f}")
    if best_state:
        model.load_state_dict(best_state)
        save_model(exp_id, seed, test_dset, best_state)
    return best_state, p_src


@torch.no_grad()
def _eval_ts_raw(
    model: TS_GIB,
    test_graph,
    feat_q_test: np.ndarray,
    device: str,
    delta_us: int,
    p_src: float,
    max_edges: int = MAX_SUB_EDGES,
    batch_size: int = BATCH_SIZE,
    max_eval_edges: int = MAX_EVAL_EDGES,
    n_jobs: int = N_JOBS,
) -> dict:
    """Evaluate TS_GIB raw-features variant on test_graph."""
    model.eval()
    src_np, dst_np, time_np = _graph_arrays(test_graph)
    labels_np = test_graph.edge_label.numpy()
    n_total   = len(labels_np)

    if n_total > max_eval_edges:
        _, eval_idx = _tts(np.arange(n_total),
                           test_size=max_eval_edges / n_total,
                           random_state=42, stratify=labels_np)
        eval_idx = np.sort(np.array(eval_idx, dtype=np.int64))
        log.info(f"  Test capped: {n_total:,} → {len(eval_idx):,}")
    else:
        eval_idx = np.arange(n_total, dtype=np.int64)

    labels_eval = labels_np[eval_idx]
    t0 = time.time()
    all_data = batch_build_subgraphs(
        src_np, dst_np, time_np,
        src_np[eval_idx], dst_np[eval_idx], time_np[eval_idx],
        delta_us=delta_us, max_edges=max_edges,
        node_feat_dim=NODE_FEAT_DIM, seed=0, n_jobs=n_jobs,
    )
    log.info(f"  Test subgraph extraction: {time.time()-t0:.1f}s")

    all_scores, all_preds = [], []
    for start in range(0, len(eval_idx), batch_size):
        ids_b   = eval_idx[start:start + batch_size]
        q_ea    = _make_q_ea_raw(ids_b, feat_q_test, device)
        dl      = all_data[start:start + batch_size]
        logits, _ = _ts_gib_forward_batch(model, dl, q_ea, device)
        all_scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        all_preds.append(logits.argmax(1).cpu().numpy())

    y_score = np.concatenate(all_scores)
    y_pred  = np.concatenate(all_preds)
    cal_mcc, _ = _calibrated_mcc(y_score, labels_eval, p_src)
    orc_mcc, _ = _oracle_mcc(y_score, labels_eval)
    try:
        auroc = float(roc_auc_score(labels_eval, y_score))
    except Exception:
        auroc = float("nan")
    metrics = compute_all_metrics(labels_eval, y_pred)
    return {
        "y_true": labels_eval, "y_pred": y_pred, "y_score": y_score,
        "reported_mcc": compute_mcc(labels_eval, y_pred),
        "calibrated_mcc": cal_mcc, "oracle_mcc": orc_mcc,
        "auroc": auroc, "auprc": metrics.get("auprc", float("nan")),
        "p_src": p_src, "macro_f1": metrics["macro_f1"],
    }


def _log_eval_results_raw(result: dict, exp_id: str, seed: int,
                           train_dsets: list, test_dset: str, elapsed: float):
    log_result(exp_id, seed, train_dsets, test_dset, "mcc",
               result["reported_mcc"], elapsed)
    log_result(exp_id, seed, train_dsets, test_dset, "calibrated_mcc",
               result["calibrated_mcc"], elapsed)
    log_result(exp_id, seed, train_dsets, test_dset, "oracle_mcc",
               result["oracle_mcc"], elapsed)
    log_result(exp_id, seed, train_dsets, test_dset, "auroc",
               result["auroc"], elapsed)
    log_result(exp_id, seed, train_dsets, test_dset, "macro_f1",
               result["macro_f1"], elapsed)
    log_result(exp_id, seed, train_dsets, test_dset, "p_src",
               result["p_src"], 0.0)
    log.info(f"  [{exp_id}] seed={seed} fold={test_dset}"
              f"  rep={result['reported_mcc']:.4f}"
              f"  cal={result['calibrated_mcc']:.4f}"
              f"  orc={result['oracle_mcc']:.4f}"
              f"  auroc={result['auroc']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# B.1.1 — E13.1: TS-GIB raw features
# ─────────────────────────────────────────────────────────────────────────────

def run_b1_ts_gib_raw(beta: float = DEFAULT_BETA_13, seeds: list = None,
                       dev: bool = True, folds: list = None):
    seeds  = seeds or [0]
    folds  = folds or ALL_FOLDS
    du     = _delta_us(DELTA_SECS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id = f"E13.1_ts_gib_raw_b{beta}"
    log.info(f"=== E13.1  TS-GIB raw features  β={beta}  seeds={seeds} ===")

    for seed in seeds:
        for fold in folds:
            train_dsets = fold["train"]
            test_dset   = fold["test"]

            if already_done(exp_id, seed, test_dset):
                log.info(f"  Skipping {exp_id} seed={seed} test={test_dset}")
                continue

            seed_everything(seed)
            t0 = time.time()
            log.info(f"\n  seed={seed}  test={test_dset}  β={beta}")

            combined, test_graph, _, _, feat_dim = _load_fold_raw(fold, dev)
            feat_q_train = combined.edge_attr_q.numpy().astype(np.float32)
            feat_q_test  = test_graph.edge_attr_q.numpy().astype(np.float32)
            log.info(f"  feat_dim={feat_dim}")

            model = TS_GIB(node_in=NODE_FEAT_DIM, ctx_edge_in=1,
                           q_edge_in=feat_dim, hidden=128,
                           use_bottleneck=True, num_domains=0)

            best_state, p_src = _train_ts_raw(
                model, combined, feat_q_train, device, seed,
                exp_id, test_dset, du, beta_max=beta,
            )

            model.eval()
            result  = _eval_ts_raw(model, test_graph, feat_q_test, device, du, p_src)
            elapsed = time.time() - t0
            _log_eval_results_raw(result, exp_id, seed, train_dsets, test_dset, elapsed)

    _print_summary(exp_id, seeds)


# ─────────────────────────────────────────────────────────────────────────────
# B.1.2 — E13.2: TS-SAGE raw features (no bottleneck)
# ─────────────────────────────────────────────────────────────────────────────

def run_b1_ts_sage_raw(seeds: list = None, dev: bool = True, folds: list = None):
    seeds  = seeds or [0]
    folds  = folds or ALL_FOLDS
    du     = _delta_us(DELTA_SECS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id = "E13.2_ts_sage_raw"
    log.info(f"=== E13.2  TS-SAGE raw features  seeds={seeds} ===")

    for seed in seeds:
        for fold in folds:
            train_dsets = fold["train"]
            test_dset   = fold["test"]

            if already_done(exp_id, seed, test_dset):
                log.info(f"  Skipping {exp_id} seed={seed} test={test_dset}")
                continue

            seed_everything(seed)
            t0 = time.time()
            log.info(f"\n  seed={seed}  test={test_dset}")

            combined, test_graph, _, _, feat_dim = _load_fold_raw(fold, dev)
            feat_q_train = combined.edge_attr_q.numpy().astype(np.float32)
            feat_q_test  = test_graph.edge_attr_q.numpy().astype(np.float32)

            # TS-SAGE = TS_GIB with no bottleneck
            model = TS_GIB(node_in=NODE_FEAT_DIM, ctx_edge_in=1,
                           q_edge_in=feat_dim, hidden=128,
                           use_bottleneck=False, num_domains=0)

            best_state, p_src = _train_ts_raw(
                model, combined, feat_q_train, device, seed,
                exp_id, test_dset, du, beta_max=0.0,
            )

            model.eval()
            result  = _eval_ts_raw(model, test_graph, feat_q_test, device, du, p_src)
            elapsed = time.time() - t0
            _log_eval_results_raw(result, exp_id, seed, train_dsets, test_dset, elapsed)

    _print_summary(exp_id, seeds)


# ─────────────────────────────────────────────────────────────────────────────
# B.1.3 — E13.3: TS-SAGE no-features, ton_iot fold (fills E8.1 gap)
# ─────────────────────────────────────────────────────────────────────────────

def run_b1_ts_sage_ton(seeds: list = None, dev: bool = True):
    """Run TS-SAGE (structure-only) on ton_iot fold using run_phase8 infra."""
    from run_phase8 import run_e8_1_ts_sage
    seeds = seeds or [0]
    ton_fold = [f for f in ALL_FOLDS if f["test"] == "ton_iot"]
    log.info(f"=== E13.3  TS-SAGE no-features ton_iot  seeds={seeds} ===")
    run_e8_1_ts_sage(seeds=seeds, delta_secs=DELTA_SECS, dev=dev, folds=ton_fold)


# ─────────────────────────────────────────────────────────────────────────────
# B.1.4 — E13.4: GIB no-features (structure-only) LODO
# ─────────────────────────────────────────────────────────────────────────────

def run_b1_gib_nofeat(beta: float = DEFAULT_BETA_13, seeds: list = None,
                       dev: bool = True, folds: list = None):
    """
    Train GIB_EGraphSAGE with structure-only edge features (edge_in=1)
    on the full LODO setup. Uses Phase 10's train_gib with use_quantile=False.
    """
    seeds  = seeds or [0]
    folds  = folds or ALL_FOLDS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id = f"E13.4_gib_nofeat_b{beta}"
    log.info(f"=== E13.4  GIB no-features  β={beta}  seeds={seeds} ===")

    from run_phase10 import _probe_gib

    for seed in seeds:
        for fold in folds:
            train_dsets = fold["train"]
            test_dset   = fold["test"]

            if already_done(exp_id, seed, test_dset):
                log.info(f"  Skipping {exp_id} seed={seed} test={test_dset}")
                continue

            seed_everything(seed)
            t0 = time.time()
            log.info(f"\n  seed={seed}  test={test_dset}  β={beta}")

            # Load with structure-only features (edge_in = 1)
            train_graphs = []
            for ds in train_dsets:
                g = load_graph(ds, tier="B", dev=dev)
                g = copy.copy(g)
                E = g.edge_attr.shape[0]
                g.edge_attr   = torch.ones(E, 1)
                g.edge_attr_q = torch.ones(E, 1)
                train_graphs.append(g)
            combined = combine_graphs(train_graphs)

            g_test = load_graph(test_dset, tier="B", dev=dev)
            g_test = copy.copy(g_test)
            E = g_test.edge_attr.shape[0]
            g_test.edge_attr   = torch.ones(E, 1)
            g_test.edge_attr_q = torch.ones(E, 1)

            val_split = _make_val_split10(combined, seed)

            model = GIB_EGraphSAGE(
                node_in=combined.x.shape[1],
                edge_in=1,
                hidden=128,
            )
            best_state = train_gib(
                model, combined, val_split, beta_max=beta,
                device=device, use_quantile=True,  # uses edge_attr_q = ones
            )
            if best_state:
                model.load_state_dict(best_state)
            save_model(exp_id, seed, test_dset, best_state or model.state_dict())

            result  = eval_egraphsage(model, g_test, device=device, use_quantile=True)
            metrics = compute_all_metrics(result["y_true"], result["y_pred"],
                                          y_true_type=g_test.edge_label_type)
            elapsed = time.time() - t0

            # Calibrated MCC
            test_scores = result["y_score"].astype(np.float32)
            vi_np  = np.array(val_split["val"], dtype=np.int64)
            p_src  = float(combined.edge_label[vi_np].numpy().mean())
            cal_m, _ = _calibrated_mcc(test_scores, result["y_true"], p_src)
            orc_m, _ = _oracle_mcc(test_scores, result["y_true"])
            try:
                from sklearn.metrics import roc_auc_score as _auroc
                auroc = float(_auroc(result["y_true"], test_scores))
            except Exception:
                auroc = float("nan")

            log.info(f"  [{exp_id}] seed={seed} fold={test_dset}"
                      f"  mcc={metrics['mcc']:.4f}  cal={cal_m:.4f}"
                      f"  orc={orc_m:.4f}  auroc={auroc:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset, "mcc",
                       metrics["mcc"], elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "macro_f1",
                       metrics["macro_f1"], elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "calibrated_mcc",
                       cal_m, elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "oracle_mcc",
                       orc_m, elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "auroc",
                       auroc, elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "p_src",
                       p_src, 0.0)

            if len(train_dsets) >= 2:
                probe = _probe_gib(model, train_dsets, tier="B", dev=dev,
                                   seed=seed, device=device, use_quantile=True)
                log.info(f"  Probe: {probe:.4f}")
                log_result(exp_id, seed, train_dsets, test_dset,
                           "dataset_probe_acc", probe, 0.0)

    _print_summary(exp_id, seeds)


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(exp_id: str, seeds: list, metric: str = "calibrated_mcc"):
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return
    fold_vals: dict = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["experiment_id"] != exp_id or row["metric"] != metric:
                continue
            if int(row["seed"]) in seeds:
                fold_vals.setdefault(row["test_dataset"], []).append(float(row["value"]))
    if not fold_vals:
        return
    log.info(f"\n  {exp_id} — {metric}:")
    for td, vals in sorted(fold_vals.items()):
        log.info(f"    {td:<20} {np.mean(vals):.4f} ± {np.std(vals):.4f}  n={len(vals)}")
    log.info(f"  mean: {np.mean([np.mean(v) for v in fold_vals.values()]):.4f}")


def _print_ablation_table(seeds: list = None):
    """Print the 2×3 ablation table (static/TS/TS+GIB × raw/nofeat)."""
    seeds = seeds or [0]
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return

    ROWS = [
        ("Static E-GS no-feat",  "E1.E_struct_only"),
        ("Static E-GS raw",      "E1.C"),
        ("Static GIB no-feat",   "E13.4_gib_nofeat_b0.01"),
        ("TS-SAGE no-feat",      "E8.1_ts_sage_d60"),
        ("TS-SAGE raw",          "E13.2_ts_sage_raw"),
        ("TS-GIB no-feat β.01",  "E12.1_ts_gib_b0.01"),
        ("TS-GIB raw β.01",      "E13.1_ts_gib_raw_b0.01"),
    ]

    data: dict = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if int(row["seed"]) not in seeds:
                continue
            for label, eid in ROWS:
                if row["experiment_id"] == eid:
                    td  = row["test_dataset"]
                    met = row["metric"]
                    data.setdefault((label, td), {})[met] = float(row["value"])

    metrics = ["calibrated_mcc", "auroc", "mcc"]
    log.info("\n  Ablation table (single-seed calibrated MCC / AUROC):")
    log.info(f"  {'Method':<28} {'lycos':>8} {'cic18':>8} {'unsw':>8} {'ton':>8}")
    for label, eid in ROWS:
        vals = []
        for td in ["lycos_ids2017", "cic_ids2018", "unsw_nb15", "ton_iot"]:
            d = data.get((label, td), {})
            v = d.get("calibrated_mcc", d.get("mcc", float("nan")))
            vals.append(f"{v:>8.3f}" if not np.isnan(v) else "     n/a")
        log.info(f"  {label:<28}" + "".join(vals))


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Phase 13: Calibration + Ablations")
    parser.add_argument("--exp", required=True,
                        choices=["a1", "a2", "a_all",
                                 "b1_ts_gib_raw", "b1_ts_sage_raw",
                                 "b1_ts_sage_ton", "b1_gib_nofeat",
                                 "all_b1", "all", "table"])
    parser.add_argument("--beta",   type=float, default=DEFAULT_BETA_13)
    parser.add_argument("--seeds",  nargs="+", type=int, default=None)
    parser.add_argument("--folds",  nargs="+", default=None,
                        help="Restrict to these test datasets")
    parser.add_argument("--target-exps", nargs="+", default=None,
                        help="Experiment IDs to analyze in A.1 (default: E12.1 + E8.1 + E1.E)")
    parser.add_argument("--no-dev", dest="dev", action="store_false")
    args = parser.parse_args()
    args.dev = getattr(args, "dev", True)

    run_folds = ALL_FOLDS
    if args.folds:
        run_folds = [f for f in ALL_FOLDS if f["test"] in args.folds]
        if not run_folds:
            log.error(f"No folds match {args.folds}"); return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.exp in ("a1", "a_all", "all"):
        run_a1_calibration_sweep(
            target_exps=args.target_exps,
            dev=args.dev,
        )

    if args.exp in ("a2", "a_all", "all"):
        run_a2_winner_analysis()

    if args.exp in ("b1_ts_gib_raw", "all_b1", "all"):
        run_b1_ts_gib_raw(beta=args.beta, seeds=args.seeds or [0],
                           dev=args.dev, folds=run_folds)

    if args.exp in ("b1_ts_sage_raw", "all_b1", "all"):
        run_b1_ts_sage_raw(seeds=args.seeds or [0], dev=args.dev, folds=run_folds)

    if args.exp in ("b1_ts_sage_ton", "all_b1", "all"):
        run_b1_ts_sage_ton(seeds=args.seeds or [0], dev=args.dev)

    if args.exp in ("b1_gib_nofeat", "all_b1", "all"):
        run_b1_gib_nofeat(beta=args.beta, seeds=args.seeds or [0],
                           dev=args.dev, folds=run_folds)

    if args.exp == "table":
        _print_ablation_table(seeds=args.seeds or [0])


if __name__ == "__main__":
    main()
