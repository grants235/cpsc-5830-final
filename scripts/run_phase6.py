#!/usr/bin/env python3
"""
Phase 6 experiments (spex6.md): reframe as anomaly detection.

Usage:
    python scripts/run_phase6.py --exp anomal_e          --seeds 0 1 2
    python scripts/run_phase6.py --exp anomal_e_msa      --seeds 0 1 2
    python scripts/run_phase6.py --exp probe             --models E6.1 E6.2
    python scripts/run_phase6.py --exp hybrid            --base best_e6 --seeds 0 1 2
    python scripts/run_phase6.py --exp per_attack        [--seed 0]
    python scripts/run_phase6.py --exp all               [--seed 0]

    Add --no-dev to use full datasets (default: dev subsamples).
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
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as _tts

from run_phase4 import (
    ALL_FOLDS, E1E_REF, E1E_MEAN,
    _make_val_split,
    _get_domain_labels, _run_probe_on_encoder,
)

from src.utils.logging import setup_logging, log_result, already_done, save_model, MODELS_DIR
from src.utils.seeding import seed_everything
from src.utils.metrics import compute_all_metrics, compute_mcc, CLASSES
from src.data.graph_builder import load_graph, combine_graphs
from src.models.egraphsage import EdgeAwareSAGE
from src.train.eval import eval_egraphsage
from src.train.train_loops import _class_weights

log = logging.getLogger(__name__)

FIGURES_DIR    = Path("results/figures/phase6")
DGI_TRAIN_MAX  = 200_000   # max benign edges for DGI training (memory)
IF_FIT_MAX     = 500_000   # max benign edges for IF fitting
DGI_BATCH      = 4096
DGI_EPOCHS     = 30
DGI_PATIENCE   = 5
CONTRAST_ALPHA = 0.5
CONTRAST_TEMP  = 0.1
N_TRIPLETS     = 512
PROJ_DIM       = 64
# Tier-A has 4 features (byte_count, packet_count, tcp_flags_any, flow_duration_ms).
# DGI needs informative edge features: with constant-ones (structure-only), the
# shuffled-feature negative equals the positive and the discriminator learns nothing.
ANOMALY_EDGE_IN = 4


# ── fold loading with informative Tier-A features ────────────────────────────

def _load_fold_anomaly(fold, dev):
    """
    Load fold with quantile-encoded Tier-A (4-dim) features.
    Must NOT use structure-only (constant 1.0) features for DGI because the
    shuffled-feature negative would be identical to the positive, making the
    discriminator's task trivial and the encoder learn nothing.
    """
    train_dsets = fold["train"]
    test_dset   = fold["test"]
    train_graphs = [load_graph(ds, tier="A", dev=dev) for ds in train_dsets]
    combined     = combine_graphs(train_graphs)
    test_graph   = load_graph(test_dset, tier="A", dev=dev)
    # Tier-A is always 4-dim so no padding needed, but guard anyway
    max_feat = combined.edge_attr.shape[1]
    d = test_graph.edge_attr.shape[1]
    if d < max_feat:
        pad = torch.zeros(test_graph.edge_attr.shape[0], max_feat - d)
        test_graph.edge_attr   = torch.cat([test_graph.edge_attr, pad], dim=1)
        test_graph.edge_attr_q = torch.cat([test_graph.edge_attr_q, pad], dim=1)
    elif d > max_feat:
        test_graph.edge_attr   = test_graph.edge_attr[:, :max_feat]
        test_graph.edge_attr_q = test_graph.edge_attr_q[:, :max_feat]
    return combined, test_graph, train_dsets, test_dset


# ── Edge DGI ─────────────────────────────────────────────────────────────────

class EdgeDGI(nn.Module):
    """
    Deep Graph Infomax adapted for edge embeddings.
    Discriminates real edge embeddings from row-shuffled-feature negatives.
    """

    def __init__(self, encoder: EdgeAwareSAGE, hidden: int = 128):
        super().__init__()
        self.encoder       = encoder
        embed_dim          = 3 * hidden
        self.discriminator = nn.Bilinear(embed_dim, embed_dim, 1)
        self.summary       = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr):
        z_pos = self.encoder.embed(x, edge_index, edge_attr)          # [E, 3H]
        perm  = torch.randperm(edge_attr.size(0), device=edge_attr.device)
        z_neg = self.encoder.embed(x, edge_index, edge_attr[perm])
        s     = self.summary(z_pos.mean(dim=0))                        # [3H]
        s_e   = s.unsqueeze(0).expand(z_pos.size(0), -1)
        score_pos = self.discriminator(z_pos, s_e).squeeze(-1)         # [E]
        score_neg = self.discriminator(z_neg, s_e.expand(z_neg.size(0), -1)).squeeze(-1)
        return score_pos, score_neg, z_pos

    @staticmethod
    def dgi_loss(score_pos, score_neg):
        return F.binary_cross_entropy_with_logits(
            torch.cat([score_pos, score_neg]),
            torch.cat([torch.ones_like(score_pos), torch.zeros_like(score_neg)])
        )


class MSA_DGI(EdgeDGI):
    """
    DGI + multi-source aware inverted contrastive loss.
    Pulls cross-dataset benign embeddings together, same-dataset apart.
    """

    def __init__(self, encoder: EdgeAwareSAGE, hidden: int = 128,
                 proj_dim: int = PROJ_DIM):
        super().__init__(encoder, hidden)
        embed_dim  = 3 * hidden
        self.proj  = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, proj_dim)
        )

    def contrastive_loss(self, z, anchor_idx, cross_idx, same_idx,
                         temp: float = CONTRAST_TEMP):
        """
        Inverted InfoNCE: cross-dataset = positive (want close),
        same-dataset = negative (want far).
        """
        z_proj  = F.normalize(self.proj(z), dim=-1)
        z_a     = z_proj[anchor_idx]
        z_c     = z_proj[cross_idx]
        z_s     = z_proj[same_idx]
        sim_c   = (z_a * z_c).sum(-1) / temp   # [T] — want high
        sim_s   = (z_a * z_s).sum(-1) / temp   # [T] — want low
        logits  = torch.stack([sim_c, sim_s], dim=1)   # [T, 2]
        targets = torch.zeros(z_a.size(0), dtype=torch.long, device=z.device)
        return F.cross_entropy(logits, targets)


# ── graph helpers ─────────────────────────────────────────────────────────────

def _filter_benign(graph):
    """Return copy of graph keeping only benign (label=0) edges."""
    idx = (graph.edge_label == 0).nonzero(as_tuple=True)[0]
    g2  = copy.copy(graph)
    g2.edge_index      = graph.edge_index[:, idx]
    g2.edge_attr       = graph.edge_attr[idx]
    g2.edge_attr_q     = graph.edge_attr_q[idx]
    g2.edge_time       = graph.edge_time[idx]
    g2.edge_label      = graph.edge_label[idx]
    g2.edge_label_type = [graph.edge_label_type[i] for i in idx.tolist()]
    return g2


def _subsample_graph(graph, max_edges, seed=0):
    """Uniformly subsample edges (preserves temporal sort)."""
    E = graph.edge_label.shape[0]
    if E <= max_edges:
        return graph
    rng = np.random.RandomState(seed)
    keep = np.sort(rng.choice(E, max_edges, replace=False))
    keep_t = torch.as_tensor(keep, dtype=torch.long)
    g2  = copy.copy(graph)
    g2.edge_index      = graph.edge_index[:, keep_t]
    g2.edge_attr       = graph.edge_attr[keep_t]
    g2.edge_attr_q     = graph.edge_attr_q[keep_t]
    g2.edge_time       = graph.edge_time[keep_t]
    g2.edge_label      = graph.edge_label[keep_t]
    g2.edge_label_type = [graph.edge_label_type[i] for i in keep.tolist()]
    return g2


def _to_local_graph(x, ei, ea, batch_ids, device):
    """Remap a batch of edge indices to a local subgraph."""
    if not isinstance(batch_ids, torch.Tensor):
        batch_ids = torch.as_tensor(batch_ids, dtype=torch.long, device=device)
    ei_b  = ei[:, batch_ids]
    ea_b  = ea[batch_ids]
    n_ids = ei_b.reshape(-1).unique()
    assoc = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    assoc[n_ids] = torch.arange(n_ids.size(0), device=x.device)
    return x[n_ids], assoc[ei_b], ea_b


@torch.no_grad()
def _extract_embeddings(encoder, graph, device, batch_size=8192):
    """Extract per-edge embeddings via local subgraph batching."""
    encoder.eval().to(device)
    x  = graph.x.to(device)
    ei = graph.edge_index.to(device)
    ea = graph.edge_attr_q.to(device)
    E  = ei.shape[1]
    parts = []
    for s in range(0, E, batch_size):
        ids = np.arange(s, min(s + batch_size, E))
        x_b, ei_b, ea_b = _to_local_graph(x, ei, ea, ids, device)
        parts.append(encoder.embed(x_b, ei_b, ea_b).cpu().numpy())
    return np.concatenate(parts)


def _sample_inverted_triplets(batch_dom, n_triplets, rng=None):
    """
    Sample (anchor, cross-dataset-idx, same-dataset-idx) within a batch.
    batch_dom: np.ndarray of domain labels for each edge in the batch.
    Returns index arrays into the batch.
    """
    if rng is None:
        rng = np.random.RandomState()
    unique = np.unique(batch_dom)
    if len(unique) < 2:
        return np.array([], int), np.array([], int), np.array([], int)

    anchors, crosses, sames = [], [], []
    dom_pools = {d: np.where(batch_dom == d)[0] for d in unique}

    attempts = 0
    while len(anchors) < n_triplets and attempts < n_triplets * 5:
        attempts += 1
        d_a = rng.choice(unique)
        d_c = rng.choice([d for d in unique if d != d_a])
        pool_a = dom_pools[d_a]
        pool_c = dom_pools[d_c]
        if len(pool_a) < 2 or len(pool_c) < 1:
            continue
        a = rng.choice(pool_a)
        c = rng.choice(pool_c)
        s = rng.choice([x for x in pool_a if x != a])
        anchors.append(a); crosses.append(c); sames.append(s)

    return np.array(anchors), np.array(crosses), np.array(sames)


# ── DGI training ─────────────────────────────────────────────────────────────

def _train_dgi_model(dgi, benign_graph, device,
                     epochs=DGI_EPOCHS, patience=DGI_PATIENCE, batch_size=DGI_BATCH):
    dgi.to(device)
    optimizer = torch.optim.AdamW(dgi.parameters(), lr=1e-3)
    x  = benign_graph.x.to(device)
    ei = benign_graph.edge_index.to(device)
    ea = benign_graph.edge_attr_q.to(device)
    E  = ei.shape[1]

    best_loss, best_state, pat_cnt = float("inf"), None, 0

    for epoch in range(epochs):
        dgi.train()
        idx_all  = np.random.permutation(E)
        ep_loss  = 0.0
        n_batches = 0
        for s in range(0, E, batch_size):
            ids = idx_all[s:s + batch_size]
            x_b, ei_b, ea_b = _to_local_graph(x, ei, ea, ids, device)
            score_pos, score_neg, _ = dgi(x_b, ei_b, ea_b)
            loss = EdgeDGI.dgi_loss(score_pos, score_neg)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(dgi.parameters(), 1.0)
            optimizer.step()
            ep_loss  += loss.item()
            n_batches += 1

        avg = ep_loss / max(n_batches, 1)
        log.info(f"  DGI epoch {epoch+1}/{epochs}  loss={avg:.4f}")

        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.cpu().clone() for k, v in dgi.state_dict().items()}
            pat_cnt    = 0
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                log.info(f"  Early stop epoch {epoch+1}")
                break

    dgi.load_state_dict(best_state)
    return best_state


def _train_msa_model(msa_dgi, benign_graph, domain_labels, device,
                     alpha=CONTRAST_ALPHA, temp=CONTRAST_TEMP,
                     epochs=DGI_EPOCHS, patience=DGI_PATIENCE, batch_size=DGI_BATCH):
    msa_dgi.to(device)
    optimizer = torch.optim.AdamW(msa_dgi.parameters(), lr=1e-3)
    x   = benign_graph.x.to(device)
    ei  = benign_graph.edge_index.to(device)
    ea  = benign_graph.edge_attr_q.to(device)
    E   = ei.shape[1]
    dom = domain_labels.numpy() if isinstance(domain_labels, torch.Tensor) else domain_labels
    rng = np.random.RandomState(0)

    best_loss, best_state, pat_cnt = float("inf"), None, 0

    for epoch in range(epochs):
        msa_dgi.train()
        idx_all   = np.random.permutation(E)
        ep_loss = ep_dgi = ep_cont = 0.0
        n_batches = 0

        for s in range(0, E, batch_size):
            ids = idx_all[s:s + batch_size]
            x_b, ei_b, ea_b = _to_local_graph(x, ei, ea, ids, device)
            score_pos, score_neg, z_b = msa_dgi(x_b, ei_b, ea_b)
            L_dgi = EdgeDGI.dgi_loss(score_pos, score_neg)

            batch_dom = dom[ids]
            a_i, c_i, s_i = _sample_inverted_triplets(batch_dom, N_TRIPLETS, rng)
            if len(a_i) > 0:
                at = torch.as_tensor(a_i, dtype=torch.long, device=device)
                ct = torch.as_tensor(c_i, dtype=torch.long, device=device)
                st = torch.as_tensor(s_i, dtype=torch.long, device=device)
                L_cont = msa_dgi.contrastive_loss(z_b, at, ct, st, temp=temp)
            else:
                L_cont = torch.tensor(0.0, device=device, requires_grad=True)

            loss = L_dgi + alpha * L_cont
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(msa_dgi.parameters(), 1.0)
            optimizer.step()
            ep_loss  += loss.item()
            ep_dgi   += L_dgi.item()
            ep_cont  += L_cont.item()
            n_batches += 1

        avg = ep_loss / max(n_batches, 1)
        log.info(f"  MSA epoch {epoch+1}/{epochs}  loss={avg:.4f}"
                 f"  dgi={ep_dgi/max(n_batches,1):.4f}"
                 f"  cont={ep_cont/max(n_batches,1):.4f}")

        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.cpu().clone() for k, v in msa_dgi.state_dict().items()}
            pat_cnt    = 0
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                log.info(f"  Early stop epoch {epoch+1}")
                break

    msa_dgi.load_state_dict(best_state)
    return best_state


# ── Isolation Forest scoring ─────────────────────────────────────────────────

def _fit_if(encoder, benign_graph, device, seed=0):
    log.info(f"  Fitting IsolationForest on {benign_graph.edge_label.shape[0]} benign edges…")
    embs  = _extract_embeddings(encoder, benign_graph, device)
    n_est = 100
    max_s = 256 if embs.shape[0] > 2_000_000 else "auto"
    # Use contamination=0.1 as a neutral default rather than "auto" which can
    # miscalibrate the decision_function offset when attack rate ≠ ~10%.
    # We bypass decision_function entirely (use score_samples + grid search),
    # so this only affects the offset attribute; set it conservatively.
    iforest = IsolationForest(n_estimators=n_est, max_samples=max_s,
                               contamination=0.1, random_state=seed)
    iforest.fit(embs)
    return iforest


def _select_threshold(if_model, encoder, val_graph, device):
    """
    Score val edges, grid-search threshold over full score range to maximise val MCC.
    Searches from 1st to 99th percentile so we don't miss the optimal cut even when
    attack rate is far from 50%.
    Returns (threshold, val_mcc_at_thresh).
    """
    embs   = _extract_embeddings(encoder, val_graph, device)
    scores = -if_model.score_samples(embs)   # higher = more anomalous
    labels = val_graph.edge_label.numpy()

    pct        = np.linspace(1.0, 99.0, 500)
    thresholds = np.unique(np.percentile(scores, pct))
    best_t, best_mcc_v = thresholds[len(thresholds) // 2], -2.0
    for t in thresholds:
        preds = (scores > t).astype(int)
        mcc   = compute_mcc(labels, preds)
        if mcc > best_mcc_v:
            best_mcc_v = mcc
            best_t     = t

    return best_t, best_mcc_v


def _anomaly_predict(if_model, encoder, graph, thresh, device):
    embs   = _extract_embeddings(encoder, graph, device)
    scores = -if_model.score_samples(embs)
    return (scores > thresh).astype(int), scores


# ── load helpers ──────────────────────────────────────────────────────────────

def _load_encoder(exp_id_prefix, seed, test_dset, node_in=8, edge_in=ANOMALY_EDGE_IN, hidden=128):
    p = MODELS_DIR / f"{exp_id_prefix}_encoder_seed{seed}_test{test_dset}.pt"
    enc = EdgeAwareSAGE(node_in=node_in, edge_in=edge_in, hidden=hidden)
    enc.load_state_dict(torch.load(p, weights_only=True))
    return enc


def _load_if(exp_id_prefix, seed, test_dset):
    p = MODELS_DIR / f"{exp_id_prefix}_if_seed{seed}_test{test_dset}.pt"
    return torch.load(p, weights_only=False)


# ── E6.1 — Anomal-E baseline ─────────────────────────────────────────────────

def run_e6_1_anomal_e(seeds, dev):
    log.info(f"=== E6.1  Anomal-E cross-dataset  seeds={seeds} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for seed in seeds:
        for fold in ALL_FOLDS:
            train_dsets = fold["train"]
            test_dset   = fold["test"]
            exp_id      = "E6.1_anomal_e"

            if already_done(exp_id, seed, test_dset):
                log.info(f"  Skipping seed={seed} test={test_dset} (already done)")
                continue

            seed_everything(seed)
            t0 = time.time()
            log.info(f"\n  Fold: train={train_dsets}  test={test_dset}  seed={seed}")

            combined, test_graph, _, _ = _load_fold_anomaly(fold, dev)
            val_split  = _make_val_split(combined)

            # Filter combined to benign only, subsample for DGI training
            benign_combined = _filter_benign(combined)
            benign_train    = _subsample_graph(benign_combined, DGI_TRAIN_MAX, seed)
            benign_if       = _subsample_graph(benign_combined, IF_FIT_MAX, seed + 1)
            log.info(f"  Benign edges: total={benign_combined.edge_label.shape[0]}"
                     f"  DGI train={benign_train.edge_label.shape[0]}"
                     f"  IF fit={benign_if.edge_label.shape[0]}")

            # Build val graph (combined, benign+attack, val indices)
            val_idx  = val_split["val"]
            val_graph = _index_subgraph(combined, val_idx)

            # Pretrain DGI on benign
            encoder = EdgeAwareSAGE(node_in=8, edge_in=ANOMALY_EDGE_IN, hidden=128)
            dgi     = EdgeDGI(encoder, hidden=128)
            dgi_state = _train_dgi_model(dgi, benign_train, device)

            # Save encoder
            enc_state = {k: v for k, v in dgi_state.items() if k.startswith("encoder.")}
            enc_state_clean = {k[len("encoder."):]: v for k, v in enc_state.items()}
            save_model(f"{exp_id}_encoder", seed, test_dset, enc_state_clean)

            # Fit IF on training benign embeddings
            dgi.load_state_dict(dgi_state)
            iforest = _fit_if(encoder, benign_if, device, seed=seed)
            save_model(f"{exp_id}_if", seed, test_dset, iforest)

            # Threshold selection on val
            thresh, val_mcc = _select_threshold(iforest, encoder, val_graph, device)
            log.info(f"  Val threshold: {thresh:.4f}  val_MCC={val_mcc:.4f}")

            # Test evaluation
            test_preds, _ = _anomaly_predict(iforest, encoder, test_graph, thresh, device)
            metrics = compute_all_metrics(
                test_graph.edge_label.numpy(), test_preds,
                y_true_type=test_graph.edge_label_type)
            elapsed = time.time() - t0

            log.info(f"  seed={seed} fold={test_dset}  MCC={metrics['mcc']:.4f}"
                     f"  macro_F1={metrics['macro_f1']:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset, "mcc", metrics["mcc"], elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "macro_f1", metrics["macro_f1"], elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "val_mcc", val_mcc, 0.0)

            # Linear probe on encoder embeddings
            probe_acc = _probe_on_encoder(encoder, train_dsets, dev, seed, device)
            log.info(f"  Probe accuracy: {probe_acc:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset, "dataset_probe_acc", probe_acc, 0.0)

    _print_method_summary("E6.1_anomal_e", seeds)


def _index_subgraph(graph, idx_list):
    """Return a subgraph with only the given edge indices."""
    idx_t = torch.as_tensor(idx_list, dtype=torch.long)
    g2    = copy.copy(graph)
    g2.edge_index      = graph.edge_index[:, idx_t]
    g2.edge_attr       = graph.edge_attr[idx_t]
    g2.edge_attr_q     = graph.edge_attr_q[idx_t]
    g2.edge_time       = graph.edge_time[idx_t]
    g2.edge_label      = graph.edge_label[idx_t]
    g2.edge_label_type = [graph.edge_label_type[i] for i in idx_list]
    return g2


def _probe_on_encoder(encoder, train_dsets, dev, seed, device, max_per_ds=10000):
    """Linear probe: predict source dataset from encoder embeddings. Returns accuracy."""
    all_embs, all_labels = [], []
    for ds_idx, ds in enumerate(train_dsets):
        g    = load_graph(ds, tier="A", dev=dev)
        embs = _extract_embeddings(encoder, g, device)
        rng  = np.random.RandomState(seed)
        idx  = rng.choice(len(embs), min(max_per_ds, len(embs)), replace=False)
        all_embs.append(embs[idx])
        all_labels.extend([ds_idx] * len(idx))
    X = np.concatenate(all_embs)
    y = np.array(all_labels)
    ti, vi = _tts(np.arange(len(X)), test_size=0.2, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
    clf.fit(X[ti], y[ti])
    return accuracy_score(y[vi], clf.predict(X[vi]))


def _print_method_summary(exp_id, seeds):
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return
    fold_vals = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["experiment_id"] != exp_id or row["metric"] != "mcc":
                continue
            if int(row["seed"]) in seeds:
                fold_vals.setdefault(row["test_dataset"], []).append(float(row["value"]))
    log.info(f"\n  {exp_id} summary:")
    fold_means = []
    for td, vals in sorted(fold_vals.items()):
        m, s = np.mean(vals), np.std(vals)
        fold_means.append(m)
        log.info(f"    {td:<20} mean={m:.4f}  std={s:.4f}  n={len(vals)}")
    if fold_means:
        overall = np.mean(fold_means)
        log.info(f"  Overall mean MCC: {overall:.4f}")
        ref = E1E_MEAN
        if overall > 0.40:
            log.info(f"  → Strong win ({overall:.3f} > 0.40). Primary positive result.")
        elif overall > 0.30:
            log.info(f"  → Real win over E1.E ({ref:.2f}). Worth writing up.")
        else:
            log.info(f"  → No improvement over E1.E. Anomaly detection alone insufficient.")


# ── E6.2 — Multi-Source Aware pretraining ────────────────────────────────────

def run_e6_2_msa(seeds, dev, alpha=CONTRAST_ALPHA):
    log.info(f"=== E6.2  Anomal-E MSA  seeds={seeds}  alpha={alpha} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for seed in seeds:
        for fold in ALL_FOLDS:
            train_dsets = fold["train"]
            test_dset   = fold["test"]
            exp_id      = "E6.2_msa"

            if already_done(exp_id, seed, test_dset):
                log.info(f"  Skipping seed={seed} test={test_dset} (already done)")
                continue

            seed_everything(seed)
            t0 = time.time()
            log.info(f"\n  Fold: train={train_dsets}  test={test_dset}  seed={seed}")

            combined, test_graph, _, _ = _load_fold_anomaly(fold, dev)
            val_split     = _make_val_split(combined)
            domain_labels = _get_domain_labels(train_dsets, dev)

            # Filter + subsample benign
            benign_combined = _filter_benign(combined)
            benign_train    = _subsample_graph(benign_combined, DGI_TRAIN_MAX, seed)
            benign_if       = _subsample_graph(benign_combined, IF_FIT_MAX, seed + 1)

            # Domain labels for the subsampled benign graph
            # _get_domain_labels returns labels aligned with the combined graph (all edges).
            # After _filter_benign, we keep only benign indices from combined.
            benign_idx_in_combined = (combined.edge_label == 0).nonzero(as_tuple=True)[0]
            dom_labels_benign      = domain_labels[benign_idx_in_combined]

            # After subsampling, take corresponding domain labels
            sub_E = benign_train.edge_label.shape[0]
            full_E_benign = benign_combined.edge_label.shape[0]
            if sub_E < full_E_benign:
                rng2 = np.random.RandomState(seed)
                sub_keep = np.sort(rng2.choice(full_E_benign, sub_E, replace=False))
                dom_labels_train = dom_labels_benign[torch.as_tensor(sub_keep)]
            else:
                dom_labels_train = dom_labels_benign

            val_graph = _index_subgraph(combined, val_split["val"])

            # Pretrain MSA-DGI
            encoder = EdgeAwareSAGE(node_in=8, edge_in=ANOMALY_EDGE_IN, hidden=128)
            msa     = MSA_DGI(encoder, hidden=128, proj_dim=PROJ_DIM)
            msa_state = _train_msa_model(
                msa, benign_train, dom_labels_train, device, alpha=alpha)

            enc_state = {k[len("encoder."):]: v
                         for k, v in msa_state.items() if k.startswith("encoder.")}
            save_model(f"{exp_id}_encoder", seed, test_dset, enc_state)

            iforest = _fit_if(encoder, benign_if, device, seed=seed)
            save_model(f"{exp_id}_if", seed, test_dset, iforest)

            thresh, val_mcc = _select_threshold(iforest, encoder, val_graph, device)
            log.info(f"  Val threshold: {thresh:.4f}  val_MCC={val_mcc:.4f}")

            test_preds, _ = _anomaly_predict(iforest, encoder, test_graph, thresh, device)
            metrics = compute_all_metrics(
                test_graph.edge_label.numpy(), test_preds,
                y_true_type=test_graph.edge_label_type)
            elapsed = time.time() - t0

            log.info(f"  seed={seed} fold={test_dset}  MCC={metrics['mcc']:.4f}"
                     f"  macro_F1={metrics['macro_f1']:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset, "mcc", metrics["mcc"], elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "macro_f1", metrics["macro_f1"], elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "val_mcc", val_mcc, 0.0)

            probe_acc = _probe_on_encoder(encoder, train_dsets, dev, seed, device)
            log.info(f"  Probe accuracy: {probe_acc:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset, "dataset_probe_acc", probe_acc, 0.0)

    _print_method_summary("E6.2_msa", seeds)
    _compare_e6("E6.1_anomal_e", "E6.2_msa", seeds)


def _compare_e6(exp_a, exp_b, seeds):
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return
    a_vals, b_vals = {}, {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["metric"] != "mcc" or int(row["seed"]) not in seeds:
                continue
            if row["experiment_id"] == exp_a:
                a_vals.setdefault(row["test_dataset"], []).append(float(row["value"]))
            elif row["experiment_id"] == exp_b:
                b_vals.setdefault(row["test_dataset"], []).append(float(row["value"]))
    log.info(f"\n  E6.1 vs E6.2 comparison:")
    diffs = []
    for td in sorted(set(a_vals) | set(b_vals)):
        va = np.mean(a_vals.get(td, [float("nan")]))
        vb = np.mean(b_vals.get(td, [float("nan")]))
        diffs.append(vb - va)
        log.info(f"    {td:<20} E6.1={va:.4f}  E6.2={vb:.4f}  Δ={vb-va:+.4f}")
    valid = [d for d in diffs if not np.isnan(d)]
    if valid:
        md = np.mean(valid)
        log.info(f"  Mean Δ E6.2 vs E6.1: {md:+.4f}")
        if md >= 0.05:
            log.info("  → E6.2 wins. Multi-source aware pretraining matters. Headline method.")
        elif md >= -0.05:
            log.info("  → E6.2 ≈ E6.1. DGI alone sufficient; E6.2 as ablation.")
        else:
            log.info("  → E6.2 underperforms. Try alpha=0.1 before giving up.")


# ── E6.3 — Linear probe diagnostic ───────────────────────────────────────────

def run_e6_3_probe(models, seed, dev):
    log.info(f"=== E6.3  Linear probe diagnostic  models={models} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_map = {"E6.1": "E6.1_anomal_e", "E6.2": "E6.2_msa"}
    baselines = {"E1.E": (0.72, 0.78), "DANN": (0.65, 0.79)}

    for model_key in models:
        exp_prefix = exp_map.get(model_key, model_key)
        log.info(f"\n  --- Model: {model_key} ({exp_prefix}) ---")

        for fold in ALL_FOLDS:
            test_dset   = fold["test"]
            train_dsets = fold["train"]
            enc_path    = MODELS_DIR / f"{exp_prefix}_encoder_seed{seed}_test{test_dset}.pt"
            if not enc_path.exists():
                log.warning(f"  Encoder not found: {enc_path}  (run E6.1/E6.2 first)")
                continue

            encoder = EdgeAwareSAGE(node_in=8, edge_in=ANOMALY_EDGE_IN, hidden=128)
            encoder.load_state_dict(torch.load(enc_path, weights_only=True))

            probe_acc = _probe_on_encoder(encoder, train_dsets, dev, seed, device)
            random_b  = 1.0 / len(train_dsets)
            log.info(f"  {model_key} fold={test_dset}  probe_acc={probe_acc:.4f}"
                     f"  (random={random_b:.2f})")

            log_result(f"E6.3_probe_{model_key}", seed, train_dsets, test_dset,
                       "dataset_probe_acc", probe_acc, 0.0)

            if probe_acc < 0.50:
                log.info("    → Invariance achieved (<50%)")
            elif probe_acc < 0.70:
                log.info("    → Partial invariance (50–70%)")
            else:
                log.info("    → Leakage persists (>70%)")

    log.info("\n  Baselines for reference:")
    for name, (lo, hi) in baselines.items():
        log.info(f"    {name}: {lo:.2f}–{hi:.2f}")


# ── E6.4 — Hybrid (anomaly score as feature for supervised model) ─────────────

def _determine_best_e6(seeds):
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return "E6.1_anomal_e"
    fold_vals = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["metric"] != "mcc" or int(row["seed"]) not in seeds:
                continue
            if row["experiment_id"] in ("E6.1_anomal_e", "E6.2_msa"):
                fold_vals.setdefault(row["experiment_id"], []).append(float(row["value"]))
    if not fold_vals:
        return "E6.1_anomal_e"
    best = max(fold_vals, key=lambda k: np.mean(fold_vals[k]))
    log.info(f"  Best E6 method: {best}  mean_mcc={np.mean(fold_vals[best]):.4f}")
    return best


def run_e6_4_hybrid(base, seeds, dev):
    log.info(f"=== E6.4  Hybrid (anomaly score + supervised)  base={base} seeds={seeds} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if base == "best_e6":
        base = _determine_best_e6(seeds)

    for seed in seeds:
        for fold in ALL_FOLDS:
            train_dsets = fold["train"]
            test_dset   = fold["test"]
            exp_id      = f"E6.4_hybrid_{base.split('_')[0]}"

            if already_done(exp_id, seed, test_dset):
                log.info(f"  Skipping seed={seed} test={test_dset} (already done)")
                continue

            # Load frozen encoder + IF for this fold
            enc_path = MODELS_DIR / f"{base}_encoder_seed{seed}_test{test_dset}.pt"
            if_path  = MODELS_DIR / f"{base}_if_seed{seed}_test{test_dset}.pt"
            if not enc_path.exists() or not if_path.exists():
                log.warning(f"  Missing {enc_path} or {if_path} — run E6.1/E6.2 first")
                continue

            seed_everything(seed)
            t0 = time.time()
            log.info(f"\n  Fold: train={train_dsets}  test={test_dset}  seed={seed}  base={base}")

            encoder = EdgeAwareSAGE(node_in=8, edge_in=ANOMALY_EDGE_IN, hidden=128)
            encoder.load_state_dict(torch.load(enc_path, weights_only=True))
            iforest = torch.load(if_path, weights_only=False)

            combined, test_graph, _, _ = _load_fold_anomaly(fold, dev)

            # Compute anomaly scores for train+test
            train_scores = _get_anomaly_scores(iforest, encoder, combined, device)
            test_scores  = _get_anomaly_scores(iforest, encoder, test_graph, device)

            # Build hybrid graphs: edge feature = [1.0, normed_score]
            train_hyb = _build_hybrid_graph(combined, train_scores)
            test_hyb  = _build_hybrid_graph(test_graph, test_scores)

            # Train supervised E-GraphSAGE on hybrid features
            from run_phase4 import _train_struct_model
            model, best_state = _train_struct_model(
                train_hyb, device, seed, epochs=30, patience=5, batch_size=2048,
                exp_id=exp_id, test_dset=test_dset,
            )

            result  = eval_egraphsage(model, test_hyb, device=device, use_quantile=True)
            metrics = compute_all_metrics(result["y_true"], result["y_pred"],
                                          y_true_type=test_hyb.edge_label_type)
            elapsed = time.time() - t0

            log.info(f"  seed={seed} fold={test_dset}  MCC={metrics['mcc']:.4f}"
                     f"  macro_F1={metrics['macro_f1']:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset, "mcc", metrics["mcc"], elapsed)
            log_result(exp_id, seed, train_dsets, test_dset, "macro_f1", metrics["macro_f1"], elapsed)

            # Probe on supervised hybrid encoder (needs 2-dim hybrid features)
            probe_acc = _probe_on_hybrid_encoder(
                model, iforest, encoder, train_dsets, dev, seed, device)
            log.info(f"  Hybrid probe accuracy: {probe_acc:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset, "dataset_probe_acc", probe_acc, 0.0)

    _print_method_summary(f"E6.4_hybrid_{base.split('_')[0]}", seeds)


def _probe_on_hybrid_encoder(sup_model, iforest, anomaly_encoder,
                             train_dsets, dev, seed, device, max_per_ds=10000):
    """
    Probe for E6.4: the supervised model takes 2-dim hybrid features, so we must
    build [1.0, anomaly_score] edge features for each training dataset before
    extracting embeddings.
    """
    all_embs, all_labels = [], []
    for ds_idx, ds in enumerate(train_dsets):
        g      = load_graph(ds, tier="A", dev=dev)
        scores = _get_anomaly_scores(iforest, anomaly_encoder, g, device)
        g_hyb  = _build_hybrid_graph(g, scores)
        embs   = _extract_embeddings(sup_model, g_hyb, device)
        rng    = np.random.RandomState(seed)
        idx    = rng.choice(len(embs), min(max_per_ds, len(embs)), replace=False)
        all_embs.append(embs[idx])
        all_labels.extend([ds_idx] * len(idx))
    X = np.concatenate(all_embs)
    y = np.array(all_labels)
    ti, vi = _tts(np.arange(len(X)), test_size=0.2, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X[ti], y[ti])
    return accuracy_score(y[vi], clf.predict(X[vi]))


def _get_anomaly_scores(iforest, encoder, graph, device):
    embs = _extract_embeddings(encoder, graph, device)
    return -iforest.score_samples(embs)    # [E] higher = more anomalous


def _build_hybrid_graph(graph, scores_np):
    """Replace edge_attr/edge_attr_q with [1.0, normalized_anomaly_score]."""
    scores_t = torch.as_tensor(scores_np, dtype=torch.float32)
    s_min, s_max = scores_t.min(), scores_t.max()
    scores_norm  = (scores_t - s_min) / (s_max - s_min + 1e-8)
    new_ea = torch.stack([torch.ones_like(scores_norm), scores_norm], dim=1)  # [E, 2]
    from src.data.graph_builder import quantile_encode
    new_ea_q = quantile_encode(new_ea)
    g2 = copy.copy(graph)
    g2.edge_attr   = new_ea
    g2.edge_attr_q = new_ea_q
    return g2


# ── E6.5 — Per-attack analysis on best E6 method ─────────────────────────────

def run_e6_5_per_attack(seed, dev):
    log.info("=== E6.5  Per-attack analysis on best E6 method ===")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    best_e6 = _determine_best_e6([seed])
    log.info(f"  Using method: {best_e6}")

    attack_classes = CLASSES[1:]
    results_table  = {}

    for fold in ALL_FOLDS:
        train_dsets = fold["train"]
        test_dset   = fold["test"]

        enc_path = MODELS_DIR / f"{best_e6}_encoder_seed{seed}_test{test_dset}.pt"
        if_path  = MODELS_DIR / f"{best_e6}_if_seed{seed}_test{test_dset}.pt"
        if not enc_path.exists() or not if_path.exists():
            log.warning(f"  Missing checkpoints for fold={test_dset} — skipping")
            continue

        encoder = EdgeAwareSAGE(node_in=8, edge_in=ANOMALY_EDGE_IN, hidden=128)
        encoder.load_state_dict(torch.load(enc_path, weights_only=True))
        iforest = torch.load(if_path, weights_only=False)

        combined, test_graph, _, _ = _load_fold_anomaly(fold, dev)
        val_split = _make_val_split(combined)
        val_graph = _index_subgraph(combined, val_split["val"])

        thresh, _ = _select_threshold(iforest, encoder, val_graph, device)
        test_preds, _ = _anomaly_predict(iforest, encoder, test_graph, thresh, device)

        metrics = compute_all_metrics(test_graph.edge_label.numpy(), test_preds,
                                      y_true_type=test_graph.edge_label_type)
        results_table[test_dset] = metrics.get("per_class_f1", {})
        log.info(f"  fold={test_dset}  MCC={metrics['mcc']:.4f}")
        for cls, f1 in metrics.get("per_class_f1", {}).items():
            log_result("E6.5_per_attack", seed, train_dsets, test_dset,
                       f"f1_{cls}", f1, 0.0)

    if not results_table:
        log.warning("  No results — run E6.1 or E6.2 first")
        return

    folds_order = [f["test"] for f in ALL_FOLDS]
    mean_f1 = {}
    for cls in attack_classes:
        vals = [results_table.get(d, {}).get(cls, float("nan")) for d in folds_order]
        mean_f1[cls] = float(np.nanmean(vals))

    sorted_cls = sorted(mean_f1, key=mean_f1.get, reverse=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(sorted_cls)), [mean_f1[c] for c in sorted_cls], color="coral")
    ax.set_xticks(range(len(sorted_cls)))
    ax.set_xticklabels(sorted_cls, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean F1 across folds")
    ax.set_ylim(0, 1)
    ax.set_title(f"E6.5  {best_e6}  per-attack mean F1")
    fig.tight_layout()
    out = FIGURES_DIR / "e6_5_per_attack_f1.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    log.info(f"  Saved {out}")

    log.info(f"\n  Per-attack mean F1:")
    for cls in sorted_cls:
        log.info(f"    {cls:<24}  {mean_f1[cls]:.4f}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Phase 6 experiments (spex6.md)")
    parser.add_argument("--exp", required=True,
                        choices=["anomal_e", "anomal_e_msa", "probe",
                                 "hybrid", "per_attack", "all"])
    parser.add_argument("--seeds",  nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seed",   type=int, default=0,
                        help="Single seed for probe / per_attack")
    parser.add_argument("--models", nargs="+", default=["E6.1", "E6.2"],
                        help="Which E6 models to probe (--exp probe)")
    parser.add_argument("--base",   default="best_e6",
                        help="Base anomaly model for hybrid: E6.1_anomal_e | E6.2_msa | best_e6")
    parser.add_argument("--alpha",  type=float, default=CONTRAST_ALPHA,
                        help="Contrastive loss weight for E6.2 (default 0.5)")
    parser.add_argument("--dev",    action="store_true", default=True)
    parser.add_argument("--no-dev", dest="dev", action="store_false")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.exp in ("anomal_e", "all"):
        run_e6_1_anomal_e(args.seeds, args.dev)

    if args.exp in ("anomal_e_msa", "all"):
        run_e6_2_msa(args.seeds, args.dev, alpha=args.alpha)

    if args.exp in ("probe", "all"):
        run_e6_3_probe(args.models, args.seed, args.dev)

    if args.exp in ("hybrid", "all"):
        run_e6_4_hybrid(args.base, args.seeds, args.dev)

    if args.exp in ("per_attack", "all"):
        run_e6_5_per_attack(args.seed, args.dev)


if __name__ == "__main__":
    main()
