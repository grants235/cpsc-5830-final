#!/usr/bin/env python3
"""
Phase 4 experiments (spex4.md): push for cross-dataset transferability.

Usage:
    python scripts/run_phase4.py --exp ensemble            [--seed 0]
    python scripts/run_phase4.py --exp dann                [--lambda_max 1.0] [--seed 0]
    python scripts/run_phase4.py --exp graphstats          [--seed 0]
    python scripts/run_phase4.py --exp dann_graphstats     [--lambda_max 1.0] [--seed 0]
    python scripts/run_phase4.py --exp all                 [--seed 0]

    Add --no-dev to use full datasets (default: dev subsamples).
"""

import argparse
import copy
import logging
import math
import sys
import time
from collections import deque, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split as _tts

from src.utils.logging import setup_logging, log_result, already_done, save_model
from src.utils.seeding import seed_everything
from src.utils.metrics import compute_all_metrics, compute_mcc
from src.data.graph_builder import load_graph, combine_graphs, quantile_encode, PROCESSED_DIR
from src.models.egraphsage import EdgeAwareSAGE
from src.train.eval import eval_egraphsage
from src.train.train_loops import _class_weights

log = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

ALL_FOLDS = [
    {"train": ["cic_ids2018",   "unsw_nb15",   "ton_iot"],       "test": "lycos_ids2017"},
    {"train": ["lycos_ids2017", "unsw_nb15",   "ton_iot"],       "test": "cic_ids2018"},
    {"train": ["lycos_ids2017", "cic_ids2018", "ton_iot"],       "test": "unsw_nb15"},
    {"train": ["lycos_ids2017", "cic_ids2018", "unsw_nb15"],     "test": "ton_iot"},
]

# E1.E reference MCC per test dataset (from phase 2/3)
E1E_REF = {
    "lycos_ids2017": -0.16,
    "cic_ids2018":    0.60,
    "unsw_nb15":      0.30,
    "ton_iot":        0.26,
}
E1E_MEAN = 0.25

FIGURES_DIR = Path("results/figures/phase4")


# ── gradient reversal ────────────────────────────────────────────────────────

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


# ── DANN model ───────────────────────────────────────────────────────────────

class DANN_EGS(nn.Module):
    """
    Domain-Adversarial E-GraphSAGE.
    encoder.embed() → z_e [E, 3H]; attack_head on z_e; domain_head on GRL(z_e).
    """

    def __init__(self, node_in: int = 8, edge_in: int = 1,
                 hidden: int = 128, num_domains: int = 3):
        super().__init__()
        self.encoder     = EdgeAwareSAGE(node_in=node_in, edge_in=edge_in,
                                         hidden=hidden, num_classes=2, dropout=0.2)
        embed_dim = 3 * hidden
        self.attack_head = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_domains),
        )

    def forward(self, x, edge_index, edge_attr, lambd: float = 0.0):
        z_e          = self.encoder.embed(x, edge_index, edge_attr)
        attack_logits = self.attack_head(z_e)
        grl_z         = GradientReversal.apply(z_e, lambd)
        domain_logits = self.domain_head(grl_z)
        return attack_logits, domain_logits, z_e

    @torch.no_grad()
    def predict(self, x, edge_index, edge_attr):
        z_e = self.encoder.embed(x, edge_index, edge_attr)
        return self.attack_head(z_e)


# ── shared data helpers ───────────────────────────────────────────────────────

def _make_struct_only(g, edge_in: int = 1):
    """Return a copy of g with edge features replaced by all-ones [E, edge_in]."""
    g = copy.copy(g)
    n = g.edge_attr.shape[0]
    g.edge_attr   = torch.ones(n, edge_in)
    g.edge_attr_q = torch.ones(n, edge_in)
    return g


def _load_fold_struct(fold, dev):
    """
    Load structure-only fold (edge_in=1).
    Returns: combined, test_graph, train_dsets, test_dset
    """
    train_dsets = fold["train"]
    test_dset   = fold["test"]

    train_graphs = [_make_struct_only(load_graph(ds, tier="B", dev=dev))
                    for ds in train_dsets]
    combined     = combine_graphs(train_graphs)
    test_graph   = _make_struct_only(load_graph(test_dset, tier="B", dev=dev))
    return combined, test_graph, train_dsets, test_dset


def _make_val_split(combined):
    n = combined.edge_label.shape[0]
    ti, vi = _tts(np.arange(n), test_size=0.2, random_state=0,
                  stratify=combined.edge_label.numpy())
    return {"train": ti.tolist(), "val": vi.tolist()}


def _get_domain_labels(train_dsets, dev):
    """
    Return domain-label tensor [E_total] aligned with combine_graphs temporal sort.
    Domain index = position of dataset in train_dsets list.
    """
    graphs = [_make_struct_only(load_graph(ds, tier="B", dev=dev))
              for ds in train_dsets]
    sizes  = [g.edge_attr.shape[0] for g in graphs]
    labels_presort = torch.cat([
        torch.full((n,), i, dtype=torch.long) for i, n in enumerate(sizes)
    ])
    # Replicate combine_graphs temporal sort
    et_cat = torch.cat([g.edge_time for g in graphs])
    order  = et_cat.argsort()
    return labels_presort[order]


@torch.no_grad()
def _get_logits(model, data, device, bs: int = 50000):
    """Return raw [E, 2] logits for all edges in data."""
    model.eval().to(device)
    x  = data.x.to(device)
    ei = data.edge_index.to(device)
    ea = data.edge_attr_q.to(device)
    parts = []
    for s in range(0, ei.shape[1], bs):
        parts.append(model(x, ei[:, s:s+bs], ea[s:s+bs]).cpu())
    return torch.cat(parts, dim=0)   # [E, 2]


def _train_struct_model(graph, device, seed, epochs=30, patience=5,
                        batch_size=2048, exp_id=None, test_dset=None):
    """Train EdgeAwareSAGE (edge_in=1) on graph. Returns (model, best_state)."""
    model = EdgeAwareSAGE(node_in=graph.x.shape[1],
                          edge_in=graph.edge_attr.shape[1],
                          hidden=128, num_classes=2, dropout=0.2)
    model.to(device)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    cw         = _class_weights(graph.edge_label, device=device)
    criterion  = nn.CrossEntropyLoss(weight=cw)

    n = graph.edge_label.shape[0]
    ti, vi = _tts(np.arange(n), test_size=0.2, random_state=seed,
                  stratify=graph.edge_label.numpy())
    ti_arr = np.array(ti, dtype=np.int64)

    x  = graph.x.to(device)
    ei = graph.edge_index.to(device)
    ea = graph.edge_attr_q.to(device)

    best_mcc, best_state, pat_cnt = -2.0, None, 0

    for epoch in range(epochs):
        model.train()
        np.random.shuffle(ti_arr)
        ep_loss = 0.0
        for s in range(0, len(ti_arr), batch_size):
            ids = ti_arr[s:s+batch_size]
            logits = model(x, ei[:, ids], ea[ids])
            loss   = criterion(logits, graph.edge_label[ids].to(device))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        model.eval()
        with torch.no_grad():
            preds = []
            for s in range(0, len(vi), batch_size):
                ids = vi[s:s+batch_size]
                preds.append(model(x, ei[:, ids], ea[ids]).argmax(1).cpu().numpy())
        val_mcc = compute_mcc(graph.edge_label[vi].numpy(), np.concatenate(preds))
        log.info(f"  epoch {epoch+1:02d}  loss={ep_loss/max(1,len(ti_arr)//batch_size):.4f}"
                 f"  val_mcc={val_mcc:.4f}")

        if val_mcc > best_mcc:
            best_mcc  = val_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt   = 0
        else:
            if epoch >= 5:
                pat_cnt += 1
                if pat_cnt >= patience:
                    log.info(f"  Early stop epoch {epoch+1}")
                    break

    log.info(f"  Best val MCC: {best_mcc:.4f}")
    if exp_id and test_dset and best_state:
        save_model(exp_id, 0, test_dset, best_state)
    model.load_state_dict(best_state)
    return model, best_state


# ── E4.1 — Single-Source Ensemble ────────────────────────────────────────────

def run_e4_1_ensemble(seed, dev):
    log.info("=== E4.1  Single-Source Ensemble ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold in ALL_FOLDS:
        train_dsets = fold["train"]
        test_dset   = fold["test"]
        exp_id      = "E4.1_ensemble"

        if already_done(exp_id, seed, test_dset):
            log.info(f"  Skipping {exp_id} test={test_dset} (already done)")
            continue

        seed_everything(seed)
        t0 = time.time()
        log.info(f"\n  Fold: train={train_dsets}  test={test_dset}")

        # Load test graph (structure-only)
        test_graph = _make_struct_only(load_graph(test_dset, tier="B", dev=dev))

        # Train one model per source dataset
        source_logits = []
        for src_ds in train_dsets:
            ckpt_path = Path(f"results/models/E4.1_{src_ds}_seed{seed}_test{test_dset}.pt")
            src_graph = _make_struct_only(load_graph(src_ds, tier="B", dev=dev))

            if ckpt_path.exists():
                log.info(f"  Loading cached model: {ckpt_path}")
                model = EdgeAwareSAGE(node_in=src_graph.x.shape[1],
                                      edge_in=1, hidden=128, num_classes=2)
                model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            else:
                log.info(f"  Training on source={src_ds}")
                model, state = _train_struct_model(
                    src_graph, device, seed,
                    epochs=30, patience=5,
                    exp_id=f"E4.1_{src_ds}", test_dset=test_dset,
                )

            logits = _get_logits(model, test_graph, device)  # [E, 2]
            source_logits.append(logits)
            log.info(f"  {src_ds}: logits shape {logits.shape}")

        # Ensemble combination
        stacked = torch.stack(source_logits, dim=0)  # [3, E, 2]
        E = stacked.shape[1]

        # (a) mean logits
        mean_logits = stacked.mean(dim=0)
        pred_mean   = mean_logits.argmax(dim=1).numpy()

        # (b) max-confidence: per edge pick model with largest |logit[1]-logit[0]|
        conf = (stacked[:, :, 1] - stacked[:, :, 0]).abs()  # [3, E]
        best_idx   = conf.argmax(dim=0)                       # [E]
        pred_maxconf = stacked[best_idx, torch.arange(E), :].argmax(dim=1).numpy()

        # (c) trimmed mean: drop most confident model per edge, average other two
        removed  = stacked[best_idx, torch.arange(E), :]      # [E, 2]
        trimmed  = (stacked.sum(dim=0) - removed) / 2.0       # [E, 2]
        pred_trim = trimmed.argmax(dim=1).numpy()

        y_true = test_graph.edge_label.numpy()
        elapsed = time.time() - t0

        for combo, preds in [("mean", pred_mean),
                              ("max_conf", pred_maxconf),
                              ("trimmed_mean", pred_trim)]:
            metrics = compute_all_metrics(y_true, preds,
                                          y_true_type=test_graph.edge_label_type)
            log.info(f"  [{combo}] fold={test_dset}  MCC={metrics['mcc']:.4f}"
                     f"  macro_F1={metrics['macro_f1']:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset,
                       f"mcc_{combo}", metrics["mcc"], elapsed)
            log_result(exp_id, seed, train_dsets, test_dset,
                       f"macro_f1_{combo}", metrics["macro_f1"], elapsed)

    # Summary
    log.info("\n  E4.1 Decision summary:")
    _print_fold_summary(exp_id, seed, metric="mcc_mean")


def _print_fold_summary(exp_id, seed, metric="mcc"):
    import csv
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return
    vals = []
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["experiment_id"] == exp_id and row["seed"] == str(seed) \
                    and row["metric"] == metric:
                vals.append(float(row["value"]))
    if vals:
        mean_mcc = np.mean(vals)
        log.info(f"  {exp_id}  metric={metric}  mean across folds: {mean_mcc:.4f}")
        if mean_mcc > E1E_MEAN + 0.10:
            log.info("  → Destructive-interference confirmed: ensemble >> E1.E by ≥0.10")
        elif mean_mcc > E1E_MEAN:
            log.info(f"  → Ensemble beats E1.E ({E1E_MEAN:.2f}) but <0.10 margin")
        else:
            log.info(f"  → Ensemble ≈ E1.E ({E1E_MEAN:.2f}); joint training not the only problem")


# ── E4.2 — DANN-EGS ──────────────────────────────────────────────────────────

def _train_dann(model, combined, domain_labels, val_split,
                device, lambda_max, epochs=50, patience=7, min_epochs=5,
                batch_size_per_domain=1024):
    model.to(device)
    optimizer       = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    cw              = _class_weights(combined.edge_label, device=device)
    attack_criterion = nn.CrossEntropyLoss(weight=cw)
    domain_criterion = nn.CrossEntropyLoss()

    x      = combined.x.to(device)
    ei     = combined.edge_index.to(device)
    ea     = combined.edge_attr_q.to(device)
    labels = combined.edge_label.to(device)
    dom_lb = domain_labels.to(device)

    train_idx = np.array(val_split["train"], dtype=np.int64)
    val_idx   = val_split["val"]

    # Group train indices by domain
    dom_train = []
    for d in range(3):
        mask = (domain_labels[train_idx] == d).numpy()
        dom_train.append(train_idx[mask].copy())
        log.info(f"  Domain {d}: {mask.sum()} train edges")

    best_mcc, best_state, pat_cnt = -2.0, None, 0

    for epoch in range(epochs):
        model.train()
        p     = (epoch + 1) / epochs
        lambd = min(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0, lambda_max)

        for d in range(3):
            np.random.shuffle(dom_train[d])

        n_batches = min(len(d) for d in dom_train) // batch_size_per_domain
        ep_cls, ep_dom = 0.0, 0.0

        for b in range(n_batches):
            s, e = b * batch_size_per_domain, (b + 1) * batch_size_per_domain
            batch_np = np.concatenate([dom_train[d][s:e] for d in range(3)])
            batch_t  = torch.as_tensor(batch_np, dtype=torch.long, device=device)

            optimizer.zero_grad()
            atk_logits, dom_logits, _ = model(x, ei[:, batch_t], ea[batch_t], lambd=lambd)
            L_cls = attack_criterion(atk_logits, labels[batch_t])
            L_dom = domain_criterion(dom_logits, dom_lb[batch_t])
            loss  = L_cls + L_dom
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_cls += L_cls.item()
            ep_dom += L_dom.item()

        avg_cls = ep_cls / max(n_batches, 1)
        avg_dom = ep_dom / max(n_batches, 1)

        # Val MCC (attack head only, lambd=0 so no GRL effect)
        model.eval()
        val_preds, val_trues = [], []
        vb_size = 10000
        with torch.no_grad():
            for s in range(0, len(val_idx), vb_size):
                vb = torch.as_tensor(val_idx[s:s+vb_size], dtype=torch.long, device=device)
                atk, _, _ = model(x, ei[:, vb], ea[vb], lambd=0.0)
                val_preds.append(atk.argmax(1).cpu().numpy())
                val_trues.append(labels[vb].cpu().numpy())
        val_mcc = compute_mcc(np.concatenate(val_trues), np.concatenate(val_preds))

        log.info(f"  DANN epoch {epoch+1:02d}  L_cls={avg_cls:.4f}  L_dom={avg_dom:.4f}"
                 f"  val_mcc={val_mcc:.4f}  lambda={lambd:.3f}")

        if val_mcc > best_mcc:
            best_mcc  = val_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt   = 0
        else:
            if epoch >= min_epochs:
                pat_cnt += 1
                if pat_cnt >= patience:
                    log.info(f"  Early stop at epoch {epoch+1}")
                    break

    log.info(f"  Best DANN val MCC: {best_mcc:.4f}")
    return best_state


def _run_probe_on_encoder(encoder, train_dsets, dev, seed, device, max_edges=10000):
    """Linear probe: predict source dataset from encoder embeddings. Returns accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    all_embs, all_labels = [], []
    for ds_idx, ds in enumerate(train_dsets):
        g    = _make_struct_only(load_graph(ds, tier="B", dev=dev))
        x_d  = g.x.to(device)
        ei_d = g.edge_index.to(device)
        ea_d = g.edge_attr_q.to(device)
        E    = ei_d.shape[1]
        embs = []
        encoder.eval().to(device)
        with torch.no_grad():
            for s in range(0, E, 50000):
                embs.append(encoder.embed(x_d, ei_d[:, s:s+50000],
                                          ea_d[s:s+50000]).cpu().numpy())
        embs_np = np.concatenate(embs)
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(embs_np), min(max_edges, len(embs_np)), replace=False)
        all_embs.append(embs_np[idx])
        all_labels.extend([ds_idx] * len(idx))

    X = np.concatenate(all_embs)
    y = np.array(all_labels)
    ti, vi = _tts(np.arange(len(X)), test_size=0.2, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
    clf.fit(X[ti], y[ti])
    acc = accuracy_score(y[vi], clf.predict(X[vi]))
    return acc


def run_e4_2_dann(lambda_max, seed, dev):
    log.info(f"=== E4.2  DANN-EGS  lambda_max={lambda_max} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold in ALL_FOLDS:
        train_dsets = fold["train"]
        test_dset   = fold["test"]
        exp_id      = "E4.2_dann"

        if already_done(exp_id, seed, test_dset):
            log.info(f"  Skipping {exp_id} test={test_dset} (already done)")
            continue

        seed_everything(seed)
        t0 = time.time()
        log.info(f"\n  Fold: train={train_dsets}  test={test_dset}")

        combined, test_graph, _, _ = _load_fold_struct(fold, dev)
        val_split    = _make_val_split(combined)
        domain_labels = _get_domain_labels(train_dsets, dev)

        model = DANN_EGS(node_in=8, edge_in=1, hidden=128, num_domains=3)
        best_state = _train_dann(
            model, combined, domain_labels, val_split, device,
            lambda_max=lambda_max, epochs=50, patience=7,
        )
        model.load_state_dict(best_state)
        save_model(exp_id, seed, test_dset, best_state)

        # Test evaluation (attack head only)
        model.eval().to(device)
        x_t  = test_graph.x.to(device)
        ei_t = test_graph.edge_index.to(device)
        ea_t = test_graph.edge_attr_q.to(device)
        E_t  = ei_t.shape[1]

        all_preds = []
        with torch.no_grad():
            for s in range(0, E_t, 50000):
                logits = model.predict(x_t, ei_t[:, s:s+50000], ea_t[s:s+50000])
                all_preds.append(logits.argmax(1).cpu().numpy())
        y_pred = np.concatenate(all_preds)
        y_true = test_graph.edge_label.numpy()

        metrics = compute_all_metrics(y_true, y_pred,
                                      y_true_type=test_graph.edge_label_type)
        elapsed = time.time() - t0
        log.info(f"  fold={test_dset}  MCC={metrics['mcc']:.4f}"
                 f"  macro_F1={metrics['macro_f1']:.4f}")
        log_result(exp_id, seed, train_dsets, test_dset,
                   "mcc", metrics["mcc"], elapsed)
        log_result(exp_id, seed, train_dsets, test_dset,
                   "macro_f1", metrics["macro_f1"], elapsed)

        # Linear probe on DANN encoder embeddings
        log.info(f"  Running linear probe on DANN encoder embeddings…")
        probe_acc = _run_probe_on_encoder(
            model.encoder, train_dsets, dev, seed, device)
        log.info(f"  Dataset-identity probe accuracy: {probe_acc:.4f}"
                 f"  (E1.E baseline ~0.72–0.78)")
        log_result(exp_id, seed, train_dsets, test_dset,
                   "dataset_probe_acc", probe_acc, 0.0)

        if probe_acc < 0.50:
            verdict = "INVARIANCE achieved (<50%)"
        elif probe_acc < 0.70:
            verdict = "PARTIAL invariance (50–70%)"
        else:
            verdict = "LEAKAGE persists (>70%)"
        log.info(f"  Probe verdict: {verdict}")

    # Summary
    log.info("\n  E4.2 Decision summary:")
    _print_fold_summary(exp_id, seed, metric="mcc")


# ── E4.3 — Graph-Statistics Edge Features ────────────────────────────────────

def _compute_graphstats(edge_index: torch.Tensor, delta: int = 1024) -> torch.Tensor:
    """
    Causal sliding-window graph-structural features for each edge.
    Features (all per-graph relative):
      0: log(1 + recent_outdeg(src))      — fan-out signal (scan)
      1: log(1 + recent_indeg(dst))       — fan-in signal (DDoS target)
      2: unique_dsts(src) / max(outdeg, 1) — dst diversity at src ∈ [0,1]
      3: pair_count(src,dst) / delta       — repeat/beaconing ratio ∈ [0,1]
    Computed before each edge is added (strictly causal).
    """
    E   = edge_index.shape[1]
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    window    = deque()              # (src, dst) in current window
    outdeg    = defaultdict(int)     # src  → # recent outgoing
    indeg     = defaultdict(int)     # dst  → # recent incoming
    dst_cnt   = defaultdict(lambda: defaultdict(int))  # src → {dst → count}
    dst_uniq  = defaultdict(int)     # src  → # distinct dsts in window
    pair_cnt  = defaultdict(int)     # (src,dst) → count in window

    feats = np.zeros((E, 4), dtype=np.float32)

    log_interval = max(E // 10, 1)
    for i in range(E):
        u, v = int(src[i]), int(dst[i])

        # Compute features BEFORE adding edge i (causal)
        feats[i, 0] = math.log1p(outdeg[u])
        feats[i, 1] = math.log1p(indeg[v])
        feats[i, 2] = dst_uniq[u] / max(outdeg[u], 1)
        feats[i, 3] = pair_cnt[(u, v)] / delta

        # Add edge i to window
        window.append((u, v))
        outdeg[u] += 1
        indeg[v]  += 1
        if dst_cnt[u][v] == 0:
            dst_uniq[u] += 1
        dst_cnt[u][v]  += 1
        pair_cnt[(u, v)] += 1

        # Evict oldest if window exceeds delta
        if len(window) > delta:
            ou, ov = window.popleft()
            outdeg[ou] -= 1
            indeg[ov]  -= 1
            dst_cnt[ou][ov] -= 1
            if dst_cnt[ou][ov] == 0:
                dst_uniq[ou] -= 1
            pair_cnt[(ou, ov)] -= 1

        if (i + 1) % log_interval == 0:
            log.info(f"    graphstats: {i+1}/{E} edges processed")

    return torch.from_numpy(feats)


def _load_or_compute_graphstats_graph(ds: str, dev: bool, delta: int = 1024):
    """Load from cache or compute graph-stats features and cache."""
    suffix = "_dev" if dev else "_full"
    cache  = PROCESSED_DIR / f"{ds}_graphstats{suffix}.pt"
    if cache.exists():
        log.info(f"  Loading cached graphstats: {cache}")
        return torch.load(cache, weights_only=False)

    log.info(f"  Computing graphstats for {ds} (delta={delta})…")
    g = load_graph(ds, tier="B", dev=dev)   # edges already in temporal order

    feats   = _compute_graphstats(g.edge_index, delta=delta)  # [E, 4]
    feats_q = quantile_encode(feats)

    g2 = copy.copy(g)
    g2.edge_attr   = feats
    g2.edge_attr_q = feats_q

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(g2, cache)
    log.info(f"  Saved graphstats cache → {cache}")
    return g2


def _load_fold_graphstats(fold, dev, delta=1024):
    train_dsets = fold["train"]
    test_dset   = fold["test"]

    train_graphs = [_load_or_compute_graphstats_graph(ds, dev, delta)
                    for ds in train_dsets]
    combined     = combine_graphs(train_graphs)
    test_graph   = _load_or_compute_graphstats_graph(test_dset, dev, delta)

    # Align feature dim (all should be 4, but be safe)
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


def run_e4_3_graphstats(seed, dev, exp_id="E4.3_graphstats"):
    log.info(f"=== E4.3  Graph-Statistics Features  exp_id={exp_id} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mccs   = {}

    for fold in ALL_FOLDS:
        train_dsets = fold["train"]
        test_dset   = fold["test"]

        if already_done(exp_id, seed, test_dset):
            log.info(f"  Skipping {exp_id} test={test_dset} (already done)")
            continue

        seed_everything(seed)
        t0 = time.time()
        log.info(f"\n  Fold: train={train_dsets}  test={test_dset}")

        combined, test_graph, _, _ = _load_fold_graphstats(fold, dev)

        model, best_state = _train_struct_model(
            combined, device, seed,
            epochs=30, patience=5, batch_size=2048,
        )
        save_model(exp_id, seed, test_dset, best_state)

        result  = eval_egraphsage(model, test_graph, device=device, use_quantile=True)
        metrics = compute_all_metrics(result["y_true"], result["y_pred"],
                                      y_true_type=test_graph.edge_label_type)
        elapsed = time.time() - t0

        log_result(exp_id, seed, train_dsets, test_dset,
                   "mcc", metrics["mcc"], elapsed)
        log_result(exp_id, seed, train_dsets, test_dset,
                   "macro_f1", metrics["macro_f1"], elapsed)

        ref  = E1E_REF.get(test_dset)
        diff = metrics["mcc"] - ref if ref is not None else 0.0
        verdict = (
            f"beats E1.E by {diff:+.3f} → run DANN combo" if diff >= 0.05
            else f"ambiguous ({diff:+.3f})" if diff >= -0.05
            else f"worse than E1.E ({diff:+.3f}) → graph-stats also dataset-specific"
        )
        log.info(f"  fold={test_dset}  MCC={metrics['mcc']:.4f}  E1.E_ref={ref}  {verdict}")
        mccs[test_dset] = metrics["mcc"]

    mean_mcc = np.mean(list(mccs.values())) if mccs else float("nan")
    log.info(f"\n  E4.3 mean MCC across computed folds: {mean_mcc:.4f}")
    if mean_mcc > E1E_MEAN + 0.05:
        log.info("  → Graph-stats have real signal; consider running dann_graphstats")
    elif mean_mcc > E1E_MEAN - 0.05:
        log.info("  → Ambiguous vs E1.E — only report if DANN combo wins")
    else:
        log.info("  → Graph-stats also dataset-specific; drop")
    return mccs


# ── E4.2 + E4.3 — DANN with graph-stats features ────────────────────────────

def run_e4_dann_graphstats(lambda_max, seed, dev):
    log.info(f"=== E4.2+E4.3  DANN with graph-stats features  lambda_max={lambda_max} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold in ALL_FOLDS:
        train_dsets = fold["train"]
        test_dset   = fold["test"]
        exp_id      = "E4.dann_graphstats"

        if already_done(exp_id, seed, test_dset):
            log.info(f"  Skipping {exp_id} test={test_dset} (already done)")
            continue

        seed_everything(seed)
        t0 = time.time()
        log.info(f"\n  Fold: train={train_dsets}  test={test_dset}")

        combined, test_graph, _, _ = _load_fold_graphstats(fold, dev)
        edge_in = combined.edge_attr.shape[1]  # 4

        # Domain labels for balanced DANN batches
        # Get graph sizes from per-dataset graphstats graphs
        gs_graphs = [_load_or_compute_graphstats_graph(ds, dev) for ds in train_dsets]
        sizes     = [g.edge_attr.shape[0] for g in gs_graphs]
        labels_pre = torch.cat([torch.full((n,), i, dtype=torch.long)
                                 for i, n in enumerate(sizes)])
        et_cat     = torch.cat([g.edge_time for g in gs_graphs])
        order      = et_cat.argsort()
        domain_labels = labels_pre[order]

        val_split = _make_val_split(combined)

        model = DANN_EGS(node_in=8, edge_in=edge_in, hidden=128, num_domains=3)
        best_state = _train_dann(
            model, combined, domain_labels, val_split, device,
            lambda_max=lambda_max, epochs=50, patience=7,
        )
        model.load_state_dict(best_state)
        save_model(exp_id, seed, test_dset, best_state)

        model.eval().to(device)
        x_t  = test_graph.x.to(device)
        ei_t = test_graph.edge_index.to(device)
        ea_t = test_graph.edge_attr_q.to(device)
        E_t  = ei_t.shape[1]

        all_preds = []
        with torch.no_grad():
            for s in range(0, E_t, 50000):
                logits = model.predict(x_t, ei_t[:, s:s+50000], ea_t[s:s+50000])
                all_preds.append(logits.argmax(1).cpu().numpy())

        y_pred  = np.concatenate(all_preds)
        y_true  = test_graph.edge_label.numpy()
        metrics = compute_all_metrics(y_true, y_pred,
                                      y_true_type=test_graph.edge_label_type)
        elapsed = time.time() - t0

        log.info(f"  fold={test_dset}  MCC={metrics['mcc']:.4f}"
                 f"  macro_F1={metrics['macro_f1']:.4f}")
        log_result(exp_id, seed, train_dsets, test_dset,
                   "mcc", metrics["mcc"], elapsed)
        log_result(exp_id, seed, train_dsets, test_dset,
                   "macro_f1", metrics["macro_f1"], elapsed)

        probe_acc = _run_probe_on_encoder(model.encoder, train_dsets, dev, seed, device)
        log.info(f"  Dataset-identity probe accuracy: {probe_acc:.4f}")
        log_result(exp_id, seed, train_dsets, test_dset,
                   "dataset_probe_acc", probe_acc, 0.0)

    log.info("\n  E4.2+E4.3 Decision summary:")
    _print_fold_summary(exp_id, seed, metric="mcc")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Phase 4 experiments (spex4.md)")
    parser.add_argument("--exp", required=True,
                        choices=["ensemble", "dann", "graphstats",
                                 "dann_graphstats", "all"])
    parser.add_argument("--lambda_max", type=float, default=1.0,
                        help="Max lambda for DANN GRL schedule (default 1.0)")
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--dev",    action="store_true", default=True)
    parser.add_argument("--no-dev", dest="dev", action="store_false")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.exp in ("ensemble", "all"):
        run_e4_1_ensemble(args.seed, args.dev)

    if args.exp in ("dann", "all"):
        run_e4_2_dann(args.lambda_max, args.seed, args.dev)

    if args.exp in ("graphstats", "all"):
        mccs = run_e4_3_graphstats(args.seed, args.dev)
        mean_mcc = np.mean(list(mccs.values())) if mccs else float("nan")
        if mean_mcc > E1E_MEAN + 0.05:
            log.info("  → E4.3 signals clear; auto-running DANN+graphstats combo")
            run_e4_dann_graphstats(args.lambda_max, args.seed, args.dev)

    if args.exp == "dann_graphstats":
        run_e4_dann_graphstats(args.lambda_max, args.seed, args.dev)


if __name__ == "__main__":
    main()
