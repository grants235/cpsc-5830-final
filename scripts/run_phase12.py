#!/usr/bin/env python3
"""
Phase 12 (spex12.md): TS-GIB — Temporal Subgraph + Variational Bottleneck
                      with calibrated evaluation.

Experiments:
  E12.1 — TS-GIB pure:              β ∈ {0.001, 0.01, 0.1}, single seed, 4 folds
  E12.2 — TS-GIB + anomaly aux:     best β from E12.1, 3 seeds × 4 folds
  E12.3 — TS-SAGE + DANN λ=0 head:  single seed, 4 folds
  E12.4 — Full combination:          best β, single seed, 4 folds
  E12.5 — Calibration ablation table for best E12 variant

Usage:
    python scripts/run_phase12.py --exp e12_1
    python scripts/run_phase12.py --exp e12_1 --betas 0.01          # single beta
    python scripts/run_phase12.py --exp e12_2 [--beta 0.01] [--seeds 0 1 2]
    python scripts/run_phase12.py --exp e12_3
    python scripts/run_phase12.py --exp e12_4 [--beta 0.01]
    python scripts/run_phase12.py --exp e12_5
    python scripts/run_phase12.py --exp all

    Add --no-dev to use full datasets (default: dev subsamples).
    Add --folds lycos_ids2017 cic_ids2018 to run on specific test folds.
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
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split as _tts
from torch_geometric.data import Batch

from run_phase4 import ALL_FOLDS, E1E_MEAN, _get_domain_labels

from src.utils.logging import setup_logging, log_result, already_done, save_model, MODELS_DIR
from src.utils.seeding import seed_everything
from src.utils.metrics import compute_all_metrics, compute_mcc
from src.data.graph_builder import load_graph, combine_graphs
from src.data.temporal_subgraph import batch_build_subgraphs
from src.models.temporal_gnn import TS_GIB
from src.models.egraphsage import EdgeAwareSAGE
from src.train.train_loops import _class_weights

log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

FIGURES_DIR    = Path("results/figures/phase12")
DELTA_SECS     = 60
BETA_SWEEP     = [0.001, 0.01, 0.1]
DEFAULT_BETA   = 0.01
BETA_WARMUP    = 5
MAX_EPOCHS     = 20
PATIENCE       = 5
BATCH_SIZE     = 2048
MAX_TRAIN_EDGES = 200_000
MAX_SUB_EDGES  = 1024
NODE_FEAT_DIM  = 8
N_JOBS         = 1

E8_1_AUROC_MEAN = 0.88   # from calibration table in spex12.md


# ── graph loading ─────────────────────────────────────────────────────────────

def _delta_us(delta_secs: int) -> int:
    return delta_secs * 1_000_000


def _graph_arrays(graph):
    return (
        graph.edge_index[0].numpy(),
        graph.edge_index[1].numpy(),
        graph.edge_time.numpy(),
    )


def _load_fold_struct(fold: dict, dev: bool):
    """Structure-only fold: edge_attr = constant 1.0 (same as E8.1)."""
    train_graphs = []
    for ds in fold["train"]:
        g = load_graph(ds, tier="B", dev=dev)
        g = copy.copy(g)
        E = g.edge_attr.shape[0]
        g.edge_attr   = torch.ones(E, 1)
        g.edge_attr_q = torch.ones(E, 1)
        train_graphs.append(g)
    combined = combine_graphs(train_graphs)

    g_test = load_graph(fold["test"], tier="B", dev=dev)
    g_test = copy.copy(g_test)
    E = g_test.edge_attr.shape[0]
    g_test.edge_attr   = torch.ones(E, 1)
    g_test.edge_attr_q = torch.ones(E, 1)
    return combined, g_test, fold["train"], fold["test"]


# ── anomaly score pre-computation (E12.2/E12.4) ──────────────────────────────

def _anomaly_scores_batched(encoder, iforest, graph, device, bs=4096):
    """Compute E6.2 anomaly scores per edge via local-subgraph batching."""
    encoder.eval().to(device)
    x  = graph.x.to(device)
    ei = graph.edge_index.to(device)
    ea = graph.edge_attr_q.to(device)
    N  = x.size(0)
    E  = ei.shape[1]
    assoc  = torch.empty(N, dtype=torch.long, device=device)
    parts  = []
    for s in range(0, E, bs):
        ei_b = ei[:, s:s + bs]
        ea_b = ea[s:s + bs]
        n_ids = ei_b.reshape(-1).unique()
        assoc.fill_(-1)
        assoc[n_ids] = torch.arange(n_ids.size(0), device=device)
        with torch.no_grad():
            emb = encoder.embed(x[n_ids], assoc[ei_b], ea_b).cpu().numpy()
        parts.append(-iforest.score_samples(emb).astype(np.float32))
        del n_ids, emb
    del assoc
    return np.concatenate(parts)


def _compute_anomaly_scores(fold: dict, seed: int, dev: bool,
                             device: str) -> tuple:
    """
    Load E6.2 encoder + IsolationForest for (seed, test_fold) and compute
    anomaly scores for every edge in the training graph and the test graph.

    Returns (scores_train [E_train], scores_test [E_test]) as float32 arrays,
    aligned with the temporal sort used by combine_graphs / load_graph.
    Returns None, None if E6.2 checkpoints are missing.
    """
    ANOMALY_EDGE_IN = 4
    test_dset = fold["test"]
    enc_path  = MODELS_DIR / f"E6.2_msa_encoder_seed{seed}_test{test_dset}.pt"
    if_path   = MODELS_DIR / f"E6.2_msa_if_seed{seed}_test{test_dset}.pt"
    if not enc_path.exists() or not if_path.exists():
        log.warning(f"  E6.2 checkpoints missing for seed={seed} test={test_dset}"
                    f" — skipping anomaly auxiliary.")
        return None, None

    log.info(f"  Loading E6.2 for anomaly scoring: seed={seed} test={test_dset}")
    encoder = EdgeAwareSAGE(node_in=8, edge_in=ANOMALY_EDGE_IN, hidden=128)
    encoder.load_state_dict(torch.load(enc_path, weights_only=True))
    iforest = torch.load(if_path, weights_only=False)

    # Training graph with FULL features (not structure-only) for score computation
    train_graphs = [load_graph(ds, tier="B", dev=dev) for ds in fold["train"]]
    combined_full = combine_graphs(train_graphs)
    # Align edge_attr_q dims to ANOMALY_EDGE_IN (same as run_phase6.py)
    d = combined_full.edge_attr_q.shape[1]
    if d < ANOMALY_EDGE_IN:
        combined_full.edge_attr_q = torch.cat(
            [combined_full.edge_attr_q,
             torch.zeros(combined_full.edge_attr_q.shape[0], ANOMALY_EDGE_IN - d)], dim=1)
    elif d > ANOMALY_EDGE_IN:
        combined_full.edge_attr_q = combined_full.edge_attr_q[:, :ANOMALY_EDGE_IN]

    scores_train = _anomaly_scores_batched(encoder, iforest, combined_full, device)
    del combined_full, train_graphs

    test_full = load_graph(test_dset, tier="B", dev=dev)
    d = test_full.edge_attr_q.shape[1]
    if d < ANOMALY_EDGE_IN:
        test_full.edge_attr_q = torch.cat(
            [test_full.edge_attr_q,
             torch.zeros(test_full.edge_attr_q.shape[0], ANOMALY_EDGE_IN - d)], dim=1)
    elif d > ANOMALY_EDGE_IN:
        test_full.edge_attr_q = test_full.edge_attr_q[:, :ANOMALY_EDGE_IN]

    scores_test = _anomaly_scores_batched(encoder, iforest, test_full, device)
    del test_full

    log.info(f"  Anomaly scores: train={len(scores_train):,}  test={len(scores_test):,}"
             f"  range=[{scores_train.min():.3f}, {scores_train.max():.3f}]")
    return scores_train, scores_test


# ── calibration helpers ───────────────────────────────────────────────────────

def _oracle_mcc(scores: np.ndarray, labels: np.ndarray, n_eval: int = 500) -> tuple:
    """Vectorised oracle MCC sweep. Returns (best_mcc, best_threshold)."""
    P  = float(labels.sum())
    N_ = float(len(labels)) - P
    if P == 0 or N_ == 0:
        return 0.0, 0.5
    E      = len(scores)
    order  = np.argsort(-scores)
    cum_tp = np.cumsum(labels[order].astype(np.float64))
    ks     = np.unique(np.round(np.linspace(0, E, n_eval + 1)).astype(np.int64))
    ks     = ks[(ks >= 0) & (ks <= E)]
    tp     = np.where(ks == 0, 0.0, cum_tp[np.clip(ks - 1, 0, E - 1)])
    tp     = np.where(ks == 0, 0.0, tp)
    fp     = ks.astype(np.float64) - tp
    tn     = N_ - fp
    fn     = P - tp
    d      = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc    = np.where(d > 0, (tp * tn - fp * fn) / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    idx    = int(np.argmax(mcc))
    k      = int(ks[idx])
    best_t = (float(scores.max()) + 1e-6 if k == 0
              else float(scores.min()) - 1e-6 if k >= E
              else float(scores[order[k]]))
    return float(mcc[idx]), best_t


def _topk_threshold(scores: np.ndarray, k: int) -> float:
    """Return threshold that sets exactly k edges positive (descending sort)."""
    if k <= 0:
        return float(scores.max()) + 1e-6
    if k >= len(scores):
        return float(scores.min()) - 1e-6
    return float(np.partition(scores, -k)[-k])


def _calibrated_mcc(scores: np.ndarray, labels: np.ndarray,
                    p_src: float) -> tuple:
    """MCC at top-k% threshold where k = p_src (source-val attack rate)."""
    k = max(1, int(round(p_src * len(scores))))
    t = _topk_threshold(scores, k)
    return compute_mcc(labels, (scores >= t).astype(int)), t


def _otsu_threshold(scores: np.ndarray) -> float:
    """Otsu's method on score histogram (256 bins)."""
    hist, edges = np.histogram(scores.astype(np.float64), bins=256)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return float(np.median(scores))
    best_t, best_var = float(np.median(scores)), -1.0
    w0 = 0.0
    mu0_sum = 0.0
    mu_total = np.sum(np.arange(256) * hist) / total
    for i in range(len(hist)):
        w0 += hist[i] / total
        mu0_sum += i * hist[i] / total
        if w0 <= 0 or w0 >= 1:
            continue
        mu0 = mu0_sum / w0
        w1  = 1.0 - w0
        mu1 = (mu_total - w0 * mu0) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var = var
            best_t   = float(edges[i + 1])
    return best_t


def _all_calibration_mccs(scores: np.ndarray, labels: np.ndarray,
                           p_src: float) -> dict:
    """
    Compute MCC under multiple calibration strategies.
    Returns dict: {method_name: mcc}.
    """
    E = len(scores)
    results = {}

    # Reported (threshold = 0.5)
    results["reported_0.5"]  = compute_mcc(labels, (scores >= 0.5).astype(int))

    # Top-k% at source-val attack rate
    mcc_cal, _ = _calibrated_mcc(scores, labels, p_src)
    results["topk_src_rate"] = mcc_cal

    # Otsu
    t_otsu = _otsu_threshold(scores)
    results["otsu"] = compute_mcc(labels, (scores >= t_otsu).astype(int))

    # GMM (2-component, subsampled)
    try:
        sub = scores
        if len(sub) > 50_000:
            rng  = np.random.RandomState(42)
            sub  = rng.choice(scores, 50_000, replace=False)
        gmm = GaussianMixture(n_components=2, n_init=1, random_state=42)
        gmm.fit(sub.reshape(-1, 1))
        means = gmm.means_.flatten()
        t_gmm = float(np.mean(sorted(means)))
        results["gmm"] = compute_mcc(labels, (scores >= t_gmm).astype(int))
    except Exception:
        results["gmm"] = float("nan")

    # Fixed top-k%
    for k_pct in [10, 20, 30]:
        k = max(1, int(round(k_pct / 100.0 * E)))
        t = _topk_threshold(scores, k)
        results[f"topk_{k_pct}pct"] = compute_mcc(labels, (scores >= t).astype(int))

    # Oracle
    orc, _ = _oracle_mcc(scores, labels)
    results["oracle"] = orc

    return results


# ── batch forward for TS_GIB ─────────────────────────────────────────────────

def _ts_gib_forward_batch(model: TS_GIB, data_list: list, q_ea: torch.Tensor,
                           device: str, use_domain: bool = False):
    """
    Build PyG Batch from data_list, run TS_GIB forward.

    Returns:
      - (attack_logits, kl) if not use_domain
      - (attack_logits, domain_logits, kl) if use_domain
    """
    batch = Batch.from_data_list([d.to(device) for d in data_list])
    ptr   = batch.ptr.to(device)
    u_globals = torch.tensor(
        [d.query_u for d in data_list], dtype=torch.long, device=device
    ) + ptr[:-1]
    v_globals = torch.tensor(
        [d.query_v for d in data_list], dtype=torch.long, device=device
    ) + ptr[:-1]

    if use_domain:
        return model.forward_with_domain(
            batch.x, batch.edge_index, batch.edge_attr,
            u_globals, v_globals, q_ea)
    return model(batch.x, batch.edge_index, batch.edge_attr,
                 u_globals, v_globals, q_ea)


@torch.no_grad()
def _ts_gib_embed_batch(model: TS_GIB, data_list: list, q_ea: torch.Tensor,
                         device: str) -> torch.Tensor:
    """Return embeddings [B, D] for the batch."""
    batch = Batch.from_data_list([d.to(device) for d in data_list])
    ptr   = batch.ptr.to(device)
    u_globals = torch.tensor(
        [d.query_u for d in data_list], dtype=torch.long, device=device
    ) + ptr[:-1]
    v_globals = torch.tensor(
        [d.query_v for d in data_list], dtype=torch.long, device=device
    ) + ptr[:-1]
    return model.embed(batch.x, batch.edge_index, batch.edge_attr,
                       u_globals, v_globals, q_ea)


def _make_q_ea(ids: np.ndarray, device: str,
               anomaly_scores: np.ndarray | None = None) -> torch.Tensor:
    """Build query-edge feature tensor for a batch of edge indices."""
    B = len(ids)
    if anomaly_scores is None:
        return torch.ones(B, 1, device=device)
    a = torch.as_tensor(anomaly_scores[ids], dtype=torch.float32, device=device)
    return torch.stack([torch.ones(B, device=device), a], dim=-1)  # [B, 2]


# ── training loop ─────────────────────────────────────────────────────────────

def _train_ts_gib(
    model: TS_GIB,
    graph,
    device: str,
    seed: int,
    exp_id: str,
    test_dset: str,
    delta_us: int,
    beta_max: float = 0.0,
    domain_labels=None,
    anomaly_scores=None,
    max_edges: int = MAX_SUB_EDGES,
    epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    max_train_edges: int = MAX_TRAIN_EDGES,
    warmup_epochs: int = BETA_WARMUP,
) -> dict:
    """
    Train TS_GIB on `graph`.  Returns best state_dict.

    domain_labels : [E_combined] tensor of source-domain integer labels (E12.3/E12.4).
    anomaly_scores: [E_combined] float32 numpy array of E6.2 scores (E12.2/E12.4).
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    cw        = _class_weights(graph.edge_label, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    dom_crit  = nn.CrossEntropyLoss() if domain_labels is not None else None

    src_np, dst_np, time_np = _graph_arrays(graph)
    labels_np = graph.edge_label.numpy()
    n         = len(labels_np)

    # Stratified 80/20 split
    ti_full, vi = _tts(np.arange(n), test_size=0.2, random_state=seed,
                       stratify=labels_np)
    ti_full = np.array(ti_full, dtype=np.int64)
    vi      = np.array(vi, dtype=np.int64)

    # Cap val so validation doesn't dominate wall time
    max_val = max_train_edges // 4
    if len(vi) > max_val:
        vi = np.random.RandomState(seed + 999).choice(vi, max_val, replace=False)

    # Source-val attack rate for calibrated evaluation
    p_src = float(labels_np[vi].mean())
    log.info(f"  train pool={len(ti_full):,}  val={len(vi):,}"
             f"  p_src(attack rate)={p_src:.4f}")

    best_mcc, best_state, pat_cnt = -2.0, None, 0
    ep_rng = np.random.RandomState(seed)

    for epoch in range(epochs):
        beta = beta_max * min(1.0, (epoch + 1) / max(1, warmup_epochs))
        model.train()

        # Stratified subsample of training edges for this epoch
        if len(ti_full) > max_train_edges:
            ep_labels = labels_np[ti_full]
            ti_arr, _ = _tts(ti_full, test_size=1.0 - max_train_edges / len(ti_full),
                              random_state=ep_rng.randint(0, 2 ** 31),
                              stratify=ep_labels)
            ti_arr = np.array(ti_arr, dtype=np.int64)
        else:
            ti_arr = ti_full.copy()
        ep_rng.shuffle(ti_arr)

        ep_loss, n_batches = 0.0, 0
        use_domain = domain_labels is not None and model.domain_head is not None

        for start in range(0, len(ti_arr), batch_size):
            ids   = ti_arr[start:start + batch_size]
            yl    = torch.as_tensor(labels_np[ids], dtype=torch.long, device=device)
            q_ea  = _make_q_ea(ids, device, anomaly_scores)

            data_list = batch_build_subgraphs(
                src_np, dst_np, time_np,
                src_np[ids], dst_np[ids], time_np[ids],
                delta_us=delta_us, max_edges=max_edges,
                node_feat_dim=NODE_FEAT_DIM, seed=seed,
                n_jobs=N_JOBS,
            )

            if use_domain:
                dom_yl = domain_labels[ids].to(device)
                atk_logits, dom_logits, kl = _ts_gib_forward_batch(
                    model, data_list, q_ea, device, use_domain=True)
                L_atk = criterion(atk_logits, yl)
                L_dom = dom_crit(dom_logits, dom_yl)
                loss  = L_atk + L_dom + beta * kl
            else:
                logits, kl = _ts_gib_forward_batch(
                    model, data_list, q_ea, device, use_domain=False)
                loss = criterion(logits, yl) + beta * kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss  += loss.item()
            n_batches += 1

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        all_preds = []
        for start in range(0, len(vi), batch_size):
            ids   = vi[start:start + batch_size]
            yl    = torch.as_tensor(labels_np[ids], dtype=torch.long, device=device)
            q_ea  = _make_q_ea(ids, device, anomaly_scores)
            with torch.no_grad():
                data_list = batch_build_subgraphs(
                    src_np, dst_np, time_np,
                    src_np[ids], dst_np[ids], time_np[ids],
                    delta_us=delta_us, max_edges=max_edges,
                    node_feat_dim=NODE_FEAT_DIM, seed=seed,
                    n_jobs=N_JOBS,
                )
                logits, _ = _ts_gib_forward_batch(
                    model, data_list, q_ea, device, use_domain=False)
            all_preds.append(logits.argmax(1).cpu().numpy())

        val_mcc = compute_mcc(labels_np[vi], np.concatenate(all_preds))
        log.info(f"  epoch {epoch + 1:02d}  β={beta:.4f}"
                 f"  loss={ep_loss / max(1, n_batches):.4f}  val_mcc={val_mcc:.4f}")

        if val_mcc > best_mcc:
            best_mcc   = val_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt    = 0
        else:
            if epoch >= 3:
                pat_cnt += 1
                if pat_cnt >= patience:
                    log.info(f"  Early stop epoch {epoch + 1}")
                    break

    log.info(f"  Best val MCC: {best_mcc:.4f}  p_src={p_src:.4f}")
    if best_state:
        model.load_state_dict(best_state)
        save_model(exp_id, seed, test_dset, best_state)
    return best_state, p_src


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_ts_gib(
    model: TS_GIB,
    graph,
    device: str,
    delta_us: int,
    p_src: float,
    anomaly_scores=None,
    max_edges: int = MAX_SUB_EDGES,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """
    Evaluate TS_GIB on graph.

    Returns dict with:
      y_true, y_pred, y_score,
      reported_mcc (thresh=0.5), calibrated_mcc (top-k%), oracle_mcc,
      calibration_methods (all strategies), auroc, auprc.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    model.eval()
    src_np, dst_np, time_np = _graph_arrays(graph)
    labels_np = graph.edge_label.numpy()
    n = len(labels_np)

    all_preds, all_scores = [], []
    for start in range(0, n, batch_size):
        ids   = np.arange(start, min(start + batch_size, n))
        q_ea  = _make_q_ea(ids, device, anomaly_scores)
        data_list = batch_build_subgraphs(
            src_np, dst_np, time_np,
            src_np[ids], dst_np[ids], time_np[ids],
            delta_us=delta_us, max_edges=max_edges,
            node_feat_dim=NODE_FEAT_DIM, seed=0,
            n_jobs=N_JOBS,
        )
        logits, _ = _ts_gib_forward_batch(model, data_list, q_ea, device)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        all_scores.append(probs)
        all_preds.append(preds)

    y_score = np.concatenate(all_scores)
    y_pred  = np.concatenate(all_preds)

    reported_mcc              = compute_mcc(labels_np, y_pred)
    calibrated_mcc, cal_t     = _calibrated_mcc(y_score, labels_np, p_src)
    oracle_mcc_val, oracle_t  = _oracle_mcc(y_score, labels_np)
    calibration_methods       = _all_calibration_mccs(y_score, labels_np, p_src)

    try:
        auroc = float(roc_auc_score(labels_np, y_score))
        auprc = float(average_precision_score(labels_np, y_score))
    except Exception:
        auroc = auprc = float("nan")

    return {
        "y_true":              labels_np,
        "y_pred":              y_pred,
        "y_score":             y_score,
        "reported_mcc":        reported_mcc,
        "calibrated_mcc":      calibrated_mcc,
        "calibrated_threshold": cal_t,
        "oracle_mcc":          oracle_mcc_val,
        "oracle_threshold":    oracle_t,
        "calibration_methods": calibration_methods,
        "auroc":               auroc,
        "auprc":               auprc,
        "p_src":               p_src,
    }


# ── linear probe ──────────────────────────────────────────────────────────────

def _probe_ts_gib(
    model: TS_GIB,
    train_dsets: list,
    dev: bool,
    seed: int,
    device: str,
    delta_us: int,
    max_per_ds: int = 500,
    anomaly_scores_per_ds=None,
) -> float:
    """Linear probe: predict source dataset from TS_GIB embeddings."""
    all_embs, all_labels = [], []
    model.eval().to(device)

    for ds_idx, ds in enumerate(train_dsets):
        g = load_graph(ds, tier="B", dev=dev)
        g = copy.copy(g)
        E = g.edge_attr.shape[0]
        g.edge_attr   = torch.ones(E, 1)
        g.edge_attr_q = torch.ones(E, 1)
        src_np, dst_np, time_np = _graph_arrays(g)

        # Anomaly scores for this individual dataset (if provided)
        anscores_ds = None
        if anomaly_scores_per_ds is not None and ds_idx < len(anomaly_scores_per_ds):
            anscores_ds = anomaly_scores_per_ds[ds_idx]

        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(E, min(max_per_ds, E), replace=False))

        embs = []
        for start in range(0, len(idx), BATCH_SIZE):
            ids       = idx[start:start + BATCH_SIZE]
            q_ea      = _make_q_ea(ids, device, anscores_ds)
            data_list = batch_build_subgraphs(
                src_np, dst_np, time_np,
                src_np[ids], dst_np[ids], time_np[ids],
                delta_us=delta_us, max_edges=MAX_SUB_EDGES,
                node_feat_dim=NODE_FEAT_DIM, seed=seed,
                n_jobs=N_JOBS,
            )
            with torch.no_grad():
                embs.append(_ts_gib_embed_batch(model, data_list, q_ea, device).cpu().numpy())

        all_embs.append(np.concatenate(embs))
        all_labels.extend([ds_idx] * sum(len(e) for e in embs[-1:]))

    X = np.concatenate(all_embs)
    y = np.array(all_labels)
    if len(np.unique(y)) < 2:
        return -1.0
    ti, vi = _tts(np.arange(len(X)), test_size=0.2, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X[ti], y[ti])
    return accuracy_score(y[vi], clf.predict(X[vi]))


# ── summary helpers ───────────────────────────────────────────────────────────

def _print_e12_summary(exp_id: str, seeds: list, metric: str = "mcc"):
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
    log.info(f"\n  {exp_id} — {metric} summary:")
    fold_means = []
    for td, vals in sorted(fold_vals.items()):
        m, s = np.mean(vals), np.std(vals)
        fold_means.append(m)
        log.info(f"    {td:<20} mean={m:.4f}  std={s:.4f}  n={len(vals)}")
    log.info(f"  Overall mean: {np.mean(fold_means):.4f}  (E1.E: {E1E_MEAN:.4f})")


def _log_eval_results(result: dict, exp_id: str, seed: int,
                       train_dsets: list, test_dset: str, elapsed: float):
    """Log all calibration metrics from an eval result dict to results.csv."""
    metrics = compute_all_metrics(result["y_true"], result["y_pred"])

    log_result(exp_id, seed, train_dsets, test_dset, "mcc",
               result["reported_mcc"], elapsed)
    log_result(exp_id, seed, train_dsets, test_dset, "calibrated_mcc",
               result["calibrated_mcc"], elapsed)
    log_result(exp_id, seed, train_dsets, test_dset, "oracle_mcc",
               result["oracle_mcc"], elapsed)
    log_result(exp_id, seed, train_dsets, test_dset, "auroc",
               result["auroc"], elapsed)
    log_result(exp_id, seed, train_dsets, test_dset, "auprc",
               result["auprc"], elapsed)
    log_result(exp_id, seed, train_dsets, test_dset, "p_src",
               result["p_src"], 0.0)
    log_result(exp_id, seed, train_dsets, test_dset, "macro_f1",
               metrics["macro_f1"], elapsed)

    # Log individual calibration strategies
    for method_name, mcc_val in result["calibration_methods"].items():
        if not np.isnan(mcc_val):
            log_result(exp_id, seed, train_dsets, test_dset,
                       f"cal_{method_name}", mcc_val, 0.0)

    log.info(f"  [{exp_id}] seed={seed} fold={test_dset}"
             f"  reported_mcc={result['reported_mcc']:.4f}"
             f"  calibrated_mcc={result['calibrated_mcc']:.4f}"
             f"  oracle_mcc={result['oracle_mcc']:.4f}"
             f"  auroc={result['auroc']:.4f}")


# ── E12.1 — TS-GIB pure: β sweep ─────────────────────────────────────────────

def run_e12_1_ts_gib_pure(betas: list = BETA_SWEEP, seeds: list = None,
                           dev: bool = True, folds: list = None):
    seeds = seeds or [0]
    folds = folds or ALL_FOLDS
    du    = _delta_us(DELTA_SECS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"=== E12.1  TS-GIB pure  β={betas}  seeds={seeds} ===")

    for beta in betas:
        exp_id = f"E12.1_ts_gib_b{beta}"
        for seed in seeds:
            for fold in folds:
                train_dsets = fold["train"]
                test_dset   = fold["test"]

                if already_done(exp_id, seed, test_dset):
                    log.info(f"  Skipping {exp_id} seed={seed} test={test_dset}")
                    continue

                seed_everything(seed)
                t0 = time.time()
                log.info(f"\n  β={beta}  seed={seed}  test={test_dset}")

                combined, test_graph, _, _ = _load_fold_struct(fold, dev)

                model = TS_GIB(node_in=NODE_FEAT_DIM, ctx_edge_in=1, q_edge_in=1,
                               hidden=128, use_bottleneck=True, num_domains=0)
                best_state, p_src = _train_ts_gib(
                    model, combined, device, seed, exp_id, test_dset,
                    delta_us=du, beta_max=beta,
                )

                model.eval()
                result  = _eval_ts_gib(model, test_graph, device, du, p_src)
                elapsed = time.time() - t0

                _log_eval_results(result, exp_id, seed, train_dsets, test_dset, elapsed)

                probe_acc = _probe_ts_gib(model, train_dsets, dev, seed, device, du)
                log.info(f"  Probe accuracy: {probe_acc:.4f}")
                log_result(exp_id, seed, train_dsets, test_dset,
                           "dataset_probe_acc", probe_acc, 0.0)

        _print_e12_summary(exp_id, seeds, "calibrated_mcc")
        _print_e12_summary(exp_id, seeds, "auroc")

    # Decision rule assessment
    _assess_e12_1_decision(betas, seeds)


def _assess_e12_1_decision(betas: list, seeds: list):
    """Read results.csv and log which β is best and what E12.2 should do."""
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return
    auroc_by_beta: dict = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["metric"] != "auroc":
                continue
            eid = row["experiment_id"]
            for beta in betas:
                if eid == f"E12.1_ts_gib_b{beta}" and int(row["seed"]) in seeds:
                    auroc_by_beta.setdefault(beta, []).append(float(row["value"]))
    if not auroc_by_beta:
        return
    log.info("\n  E12.1 decision rule:")
    for beta in betas:
        vals = auroc_by_beta.get(beta, [])
        mean = np.mean(vals) if vals else float("nan")
        log.info(f"    β={beta:<6} mean AUROC={mean:.4f}  n_folds={len(vals)}")
    best_beta = max(auroc_by_beta, key=lambda b: np.mean(auroc_by_beta[b]))
    best_mean = np.mean(auroc_by_beta[best_beta])
    log.info(f"  Best β: {best_beta}  mean AUROC={best_mean:.4f}")
    if best_mean > 0.85:
        log.info("  → AUROC > 0.85: proceed to E12.2 with this β.")
    elif best_mean >= 0.80:
        log.info("  → AUROC 0.80-0.85: bottleneck contributes; proceed to E12.2.")
    else:
        log.info("  → AUROC < 0.80: bottleneck damages TS-SAGE; pivot to E12.3.")


def _get_best_beta_e12() -> float:
    """Return β with highest mean AUROC from E12.1 results."""
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return DEFAULT_BETA
    auroc_by_beta: dict = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["metric"] != "auroc":
                continue
            for beta in BETA_SWEEP:
                if row["experiment_id"] == f"E12.1_ts_gib_b{beta}":
                    auroc_by_beta.setdefault(beta, []).append(float(row["value"]))
    if not auroc_by_beta:
        log.warning("  No E12.1 results; defaulting to β=0.01")
        return DEFAULT_BETA
    best = max(auroc_by_beta, key=lambda b: np.mean(auroc_by_beta[b]))
    log.info(f"  Best β from E12.1: {best}")
    return float(best)


# ── E12.2 — TS-GIB + anomaly auxiliary ───────────────────────────────────────

def run_e12_2_ts_gib_anomaly(beta: float = None, seeds: list = None,
                              dev: bool = True, folds: list = None):
    if beta is None:
        beta = _get_best_beta_e12()
    seeds = seeds or [0, 1, 2]
    folds = folds or ALL_FOLDS
    du    = _delta_us(DELTA_SECS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id = f"E12.2_ts_gib_anomaly_b{beta}"
    log.info(f"=== E12.2  TS-GIB + anomaly  β={beta}  seeds={seeds} ===")

    for seed in seeds:
        for fold in folds:
            train_dsets = fold["train"]
            test_dset   = fold["test"]

            if already_done(exp_id, seed, test_dset):
                log.info(f"  Skipping {exp_id} seed={seed} test={test_dset}")
                continue

            # Load anomaly scores (seed-matched)
            scores_train, scores_test = _compute_anomaly_scores(fold, seed, dev, device)
            if scores_train is None:
                log.warning(f"  Skipping fold (no E6.2 checkpoints).")
                continue

            seed_everything(seed)
            t0 = time.time()
            log.info(f"\n  seed={seed}  test={test_dset}  β={beta}")

            combined, test_graph, _, _ = _load_fold_struct(fold, dev)

            model = TS_GIB(node_in=NODE_FEAT_DIM, ctx_edge_in=1, q_edge_in=2,
                           hidden=128, use_bottleneck=True, num_domains=0)
            best_state, p_src = _train_ts_gib(
                model, combined, device, seed, exp_id, test_dset,
                delta_us=du, beta_max=beta,
                anomaly_scores=scores_train,
            )

            model.eval()
            result  = _eval_ts_gib(model, test_graph, device, du, p_src,
                                    anomaly_scores=scores_test)
            elapsed = time.time() - t0

            _log_eval_results(result, exp_id, seed, train_dsets, test_dset, elapsed)

            probe_acc = _probe_ts_gib(model, train_dsets, dev, seed, device, du)
            log.info(f"  Probe accuracy: {probe_acc:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset,
                       "dataset_probe_acc", probe_acc, 0.0)

    _print_e12_summary(exp_id, seeds, "calibrated_mcc")
    _print_e12_summary(exp_id, seeds, "auroc")
    _assess_e12_2_decision(exp_id, seeds)


def _assess_e12_2_decision(exp_id: str, seeds: list):
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return
    cal_mccs, aurocs = [], []
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["experiment_id"] != exp_id or int(row["seed"]) not in seeds:
                continue
            if row["metric"] == "calibrated_mcc":
                cal_mccs.append(float(row["value"]))
            elif row["metric"] == "auroc":
                aurocs.append(float(row["value"]))
    if not cal_mccs:
        return
    mean_cal = np.mean(cal_mccs)
    std_cal  = np.std(cal_mccs)
    mean_aur = np.mean(aurocs) if aurocs else float("nan")
    log.info(f"\n  E12.2 decision:")
    log.info(f"    mean calibrated MCC={mean_cal:.4f} ± {std_cal:.4f}  AUROC={mean_aur:.4f}")
    if mean_aur > 0.85 and mean_cal > 0.45 and std_cal < 0.05:
        log.info("  → Robust positive result. Headline method.")
    elif mean_aur > 0.85 and std_cal >= 0.05:
        log.info("  → Architecture works, calibration unstable. Report with caveat.")
    else:
        log.info("  → Revert to E12.1 best variant as main result.")


# ── E12.3 — TS-SAGE + DANN λ=0 auxiliary head ────────────────────────────────

def run_e12_3_ts_dann_aux(seeds: list = None, dev: bool = True, folds: list = None):
    seeds = seeds or [0]
    folds = folds or ALL_FOLDS
    du    = _delta_us(DELTA_SECS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id = "E12.3_ts_dann_no_grl"
    log.info(f"=== E12.3  TS-SAGE + DANN λ=0  seeds={seeds} ===")

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

            combined, test_graph, _, _ = _load_fold_struct(fold, dev)
            # Domain labels aligned with combined graph temporal sort
            domain_labels = _get_domain_labels(train_dsets, dev)

            model = TS_GIB(node_in=NODE_FEAT_DIM, ctx_edge_in=1, q_edge_in=1,
                           hidden=128, use_bottleneck=False,
                           num_domains=len(train_dsets))
            best_state, p_src = _train_ts_gib(
                model, combined, device, seed, exp_id, test_dset,
                delta_us=du, beta_max=0.0,
                domain_labels=domain_labels,
            )

            model.eval()
            result  = _eval_ts_gib(model, test_graph, device, du, p_src)
            elapsed = time.time() - t0

            _log_eval_results(result, exp_id, seed, train_dsets, test_dset, elapsed)

            probe_acc = _probe_ts_gib(model, train_dsets, dev, seed, device, du)
            log.info(f"  Probe accuracy: {probe_acc:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset,
                       "dataset_probe_acc", probe_acc, 0.0)

    _print_e12_summary(exp_id, seeds, "calibrated_mcc")
    _print_e12_summary(exp_id, seeds, "auroc")
    _compare_e12_3_to_baseline(seeds)


def _compare_e12_3_to_baseline(seeds):
    """Compare E12.3 vs best E12.1 β."""
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return
    e12_3_mccs: dict = {}
    e12_1_mccs: dict = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["metric"] != "calibrated_mcc" or int(row["seed"]) not in seeds:
                continue
            td  = row["test_dataset"]
            eid = row["experiment_id"]
            if eid == "E12.3_ts_dann_no_grl":
                e12_3_mccs.setdefault(td, []).append(float(row["value"]))
            elif eid.startswith("E12.1_ts_gib_b"):
                e12_1_mccs.setdefault(td, []).append(float(row["value"]))

    if not e12_3_mccs or not e12_1_mccs:
        return
    log.info("\n  E12.3 vs best E12.1 (calibrated MCC):")
    for td in sorted(set(e12_3_mccs) | set(e12_1_mccs)):
        v3 = np.mean(e12_3_mccs.get(td, [float("nan")]))
        v1 = max(e12_1_mccs.get(td, [float("nan")]))
        log.info(f"    {td:<20} E12.3={v3:.4f}  E12.1_best={v1:.4f}  Δ={v3-v1:+.4f}")


# ── E12.4 — Full combination ──────────────────────────────────────────────────

def run_e12_4_full(beta: float = None, seeds: list = None,
                   dev: bool = True, folds: list = None):
    if beta is None:
        beta = _get_best_beta_e12()
    seeds = seeds or [0]
    folds = folds or ALL_FOLDS
    du    = _delta_us(DELTA_SECS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id = f"E12.4_ts_gib_full_b{beta}"
    log.info(f"=== E12.4  Full combination  β={beta}  seeds={seeds} ===")

    for seed in seeds:
        for fold in folds:
            train_dsets = fold["train"]
            test_dset   = fold["test"]

            if already_done(exp_id, seed, test_dset):
                log.info(f"  Skipping {exp_id} seed={seed} test={test_dset}")
                continue

            scores_train, scores_test = _compute_anomaly_scores(fold, seed, dev, device)
            if scores_train is None:
                log.warning(f"  Skipping fold (no E6.2 checkpoints).")
                continue

            seed_everything(seed)
            t0 = time.time()
            log.info(f"\n  seed={seed}  test={test_dset}  β={beta}")

            combined, test_graph, _, _ = _load_fold_struct(fold, dev)
            domain_labels = _get_domain_labels(train_dsets, dev)

            model = TS_GIB(node_in=NODE_FEAT_DIM, ctx_edge_in=1, q_edge_in=2,
                           hidden=128, use_bottleneck=True,
                           num_domains=len(train_dsets))
            best_state, p_src = _train_ts_gib(
                model, combined, device, seed, exp_id, test_dset,
                delta_us=du, beta_max=beta,
                domain_labels=domain_labels,
                anomaly_scores=scores_train,
            )

            model.eval()
            result  = _eval_ts_gib(model, test_graph, device, du, p_src,
                                    anomaly_scores=scores_test)
            elapsed = time.time() - t0

            _log_eval_results(result, exp_id, seed, train_dsets, test_dset, elapsed)

            probe_acc = _probe_ts_gib(model, train_dsets, dev, seed, device, du)
            log.info(f"  Probe accuracy: {probe_acc:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset,
                       "dataset_probe_acc", probe_acc, 0.0)

    _print_e12_summary(exp_id, seeds, "calibrated_mcc")
    _print_e12_summary(exp_id, seeds, "auroc")
    _compare_e12_all(seeds)


def _compare_e12_all(seeds):
    """Print a comparison table of all E12 variants."""
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return
    variants = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["metric"] not in ("calibrated_mcc", "auroc"):
                continue
            if int(row["seed"]) not in seeds:
                continue
            eid = row["experiment_id"]
            if not eid.startswith("E12."):
                continue
            variants.setdefault(eid, {}).setdefault(row["metric"], []).append(
                float(row["value"]))

    if not variants:
        return
    log.info("\n  E12 comparison (all variants):")
    log.info(f"  {'Experiment':<40} {'cal_MCC':>8} {'AUROC':>8}")
    for eid in sorted(variants):
        cal = variants[eid].get("calibrated_mcc", [])
        aur = variants[eid].get("auroc", [])
        m_cal = np.mean(cal) if cal else float("nan")
        m_aur = np.mean(aur) if aur else float("nan")
        log.info(f"  {eid:<40} {m_cal:>8.4f} {m_aur:>8.4f}")


# ── E12.5 — Calibration ablation table ───────────────────────────────────────

def run_e12_5_calibration_table(dev: bool = True, seeds: list = None,
                                 out_csv: str = "results/e12_calibration_ablation.csv"):
    """
    Build calibration ablation table for all E12 variants.
    Reads calibrated_mcc metrics stored under cal_* keys from results.csv.
    """
    seeds = seeds or [0]
    results_path = Path("results/results.csv")
    out_path     = Path(out_csv)

    if not results_path.exists():
        log.warning("  No results.csv — run E12.1-E12.4 first.")
        return

    # Collect all cal_* metrics from results.csv
    rows_by_exp: dict = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            eid = row["experiment_id"]
            if not eid.startswith("E12."):
                continue
            if int(row["seed"]) not in seeds:
                continue
            td     = row["test_dataset"]
            metric = row["metric"]
            val    = float(row["value"])
            rows_by_exp.setdefault((eid, td), {})[metric] = val

    if not rows_by_exp:
        log.warning("  No E12 results found in results.csv.")
        return

    # Build table rows
    cal_methods = ["reported_0.5", "topk_src_rate", "otsu", "gmm",
                   "topk_10pct", "topk_20pct", "topk_30pct", "oracle"]
    fieldnames  = ["experiment_id", "test_fold"] + cal_methods + ["auroc", "probe_acc"]
    out_rows    = []

    for (eid, td), metrics in sorted(rows_by_exp.items()):
        r = {"experiment_id": eid, "test_fold": td}
        r["reported_0.5"]   = metrics.get("cal_reported_0.5", metrics.get("mcc", ""))
        r["topk_src_rate"]  = metrics.get("cal_topk_src_rate",
                                           metrics.get("calibrated_mcc", ""))
        r["otsu"]           = metrics.get("cal_otsu", "")
        r["gmm"]            = metrics.get("cal_gmm", "")
        r["topk_10pct"]     = metrics.get("cal_topk_10pct", "")
        r["topk_20pct"]     = metrics.get("cal_topk_20pct", "")
        r["topk_30pct"]     = metrics.get("cal_topk_30pct", "")
        r["oracle"]         = metrics.get("oracle_mcc", "")
        r["auroc"]          = metrics.get("auroc", "")
        r["probe_acc"]      = metrics.get("dataset_probe_acc", "")
        out_rows.append(r)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    log.info(f"\n  E12.5 calibration ablation → {out_path}  ({len(out_rows)} rows)")

    # Print summary
    log.info(f"\n  {'Experiment':<42} {'rep':>6} {'topk':>6} {'otsu':>6}"
             f" {'gmm':>6} {'orc':>6} {'auroc':>6}")
    for r in out_rows:
        def _fmt(v):
            try:
                return f"{float(v):>6.3f}"
            except (ValueError, TypeError):
                return "   n/a"
        log.info(f"  {r['experiment_id']:<42}"
                 f" {_fmt(r['reported_0.5'])}"
                 f" {_fmt(r['topk_src_rate'])}"
                 f" {_fmt(r['otsu'])}"
                 f" {_fmt(r['gmm'])}"
                 f" {_fmt(r['oracle'])}"
                 f" {_fmt(r['auroc'])}")


# ── AUROC evaluation ──────────────────────────────────────────────────────────

def _get_auroc_from_results(exp_id: str, seeds: list) -> float:
    results_path = Path("results/results.csv")
    vals = []
    if results_path.exists():
        with open(results_path) as f:
            for row in csv.DictReader(f):
                if (row["experiment_id"] == exp_id and row["metric"] == "auroc"
                        and int(row["seed"]) in seeds):
                    vals.append(float(row["value"]))
    return np.mean(vals) if vals else float("nan")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Phase 12: TS-GIB")
    parser.add_argument("--exp", choices=["e12_1", "e12_2", "e12_3", "e12_4",
                                           "e12_5", "all"],
                        required=True)
    parser.add_argument("--betas", nargs="+", type=float, default=BETA_SWEEP,
                        help="β values for E12.1 (default: 0.001 0.01 0.1)")
    parser.add_argument("--beta", type=float, default=None,
                        help="Single β for E12.2/E12.4 (default: best from E12.1)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Seeds to run (default: [0] for E12.1/3/4, [0,1,2] for E12.2)")
    parser.add_argument("--folds", nargs="+", default=None,
                        help="Test datasets to evaluate on (default: all 4)")
    parser.add_argument("--no-dev", dest="dev", action="store_false",
                        help="Use full datasets (no dev subsampling)")
    args = parser.parse_args()
    args.dev = getattr(args, "dev", True)

    # Filter folds if requested
    run_folds = ALL_FOLDS
    if args.folds:
        run_folds = [f for f in ALL_FOLDS if f["test"] in args.folds]
        if not run_folds:
            log.error(f"No folds match {args.folds}")
            sys.exit(1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.exp in ("e12_1", "all"):
        run_e12_1_ts_gib_pure(
            betas=args.betas,
            seeds=args.seeds or [0],
            dev=args.dev,
            folds=run_folds,
        )

    if args.exp in ("e12_2", "all"):
        run_e12_2_ts_gib_anomaly(
            beta=args.beta,
            seeds=args.seeds or [0, 1, 2],
            dev=args.dev,
            folds=run_folds,
        )

    if args.exp in ("e12_3", "all"):
        run_e12_3_ts_dann_aux(
            seeds=args.seeds or [0],
            dev=args.dev,
            folds=run_folds,
        )

    if args.exp in ("e12_4", "all"):
        run_e12_4_full(
            beta=args.beta,
            seeds=args.seeds or [0],
            dev=args.dev,
            folds=run_folds,
        )

    if args.exp in ("e12_5", "all"):
        run_e12_5_calibration_table(
            dev=args.dev,
            seeds=args.seeds or [0, 1, 2],
        )


if __name__ == "__main__":
    main()
