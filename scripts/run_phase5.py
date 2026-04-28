#!/usr/bin/env python3
"""
Phase 5 experiments (spex5.md): improve and solidify DANN-EGS.

Usage:
    python scripts/run_phase5.py --exp dann           --seeds 1 2
    python scripts/run_phase5.py --exp dann_no_grl    [--seed 0]
    python scripts/run_phase5.py --exp source_gated   [--seed 0]
    python scripts/run_phase5.py --exp cdan           [--seed 0]
    python scripts/run_phase5.py --exp dropedge       --p_drop 0.2 0.5 [--base dann] [--seed 0]
    python scripts/run_phase5.py --exp all            [--seed 0]

    Add --no-dev to use full datasets (default: dev subsamples).
"""

import argparse
import copy
import csv
import logging
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root
sys.path.insert(0, str(Path(__file__).resolve().parent))      # scripts dir

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split as _tts

# Import reusable pieces from run_phase4
from run_phase4 import (
    GradientReversal, DANN_EGS,
    ALL_FOLDS, E1E_REF, E1E_MEAN,
    _make_struct_only, _load_fold_struct, _make_val_split,
    _get_domain_labels, _train_dann, _run_probe_on_encoder,
    _get_logits, _train_struct_model, _print_fold_summary,
)

from src.utils.logging import setup_logging, log_result, already_done, save_model
from src.utils.seeding import seed_everything
from src.utils.metrics import compute_all_metrics, compute_mcc
from src.data.graph_builder import load_graph, combine_graphs, quantile_encode, PROCESSED_DIR
from src.models.egraphsage import EdgeAwareSAGE
from src.train.eval import eval_egraphsage
from src.train.train_loops import _class_weights

log = logging.getLogger(__name__)

FIGURES_DIR = Path("results/figures/phase5")

# ── CDAN-EGS (class-conditional DANN) ────────────────────────────────────────

class CDAN_EGS(nn.Module):
    """
    DANN with class-conditional domain alignment.
    Domain classifier input: outer product of z_e and attack probs → [E, embed_dim * num_classes].
    """

    def __init__(self, node_in: int = 8, edge_in: int = 1,
                 hidden: int = 128, num_domains: int = 3, num_classes: int = 2):
        super().__init__()
        self.encoder     = EdgeAwareSAGE(node_in=node_in, edge_in=edge_in,
                                         hidden=hidden, num_classes=2, dropout=0.2)
        embed_dim  = 3 * hidden           # 384
        cdan_dim   = embed_dim * num_classes  # 768

        self.attack_head = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(cdan_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_domains),
        )

    def forward(self, x, edge_index, edge_attr, lambd: float = 0.0):
        z_e          = self.encoder.embed(x, edge_index, edge_attr)
        attack_logits = self.attack_head(z_e)

        # Class-conditional outer product (detach attack_probs to avoid interference)
        attack_probs = F.softmax(attack_logits, dim=-1).detach()           # [E, 2]
        joint        = torch.einsum('bi,bj->bij', z_e, attack_probs).flatten(1)  # [E, 768]
        grl_joint    = GradientReversal.apply(joint, lambd)
        domain_logits = self.domain_head(grl_joint)
        return attack_logits, domain_logits, z_e

    @torch.no_grad()
    def predict(self, x, edge_index, edge_attr):
        z_e = self.encoder.embed(x, edge_index, edge_attr)
        return self.attack_head(z_e)


# ── gate network for source-conditional gating ───────────────────────────────

class GateNetwork(nn.Module):
    """2-layer MLP: Tier-A 4 features → softmax over num_sources."""

    def __init__(self, in_features: int = 4, hidden: int = 64, num_sources: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_sources),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)  # [E, num_sources]


# ── source-specialist DANN training ─────────────────────────────────────────

def _train_source_specialist(
    model,          # DANN_EGS
    combined,       # combined struct-only graph (all 3 sources)
    domain_labels,  # [E] domain index per edge (aligned with combined)
    src_domain,     # which domain is the "primary" for attack classification
    device,
    lambda_max=1.0,
    epochs=50,
    patience=7,
    min_epochs=5,
    batch_size=1024,
):
    """
    Train a per-source DANN_EGS specialist.
    Attack loss:  computed on src_domain edges only.
    Domain loss:  computed on balanced batches from all 3 domains.
    Early stop:   on src_domain val-MCC.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    src_mask   = (domain_labels == src_domain).numpy()
    src_labels = combined.edge_label[src_mask]
    cw         = _class_weights(src_labels, device=device)
    atk_crit   = nn.CrossEntropyLoss(weight=cw)
    dom_crit   = nn.CrossEntropyLoss()

    x      = combined.x.to(device)
    ei     = combined.edge_index.to(device)
    ea     = combined.edge_attr_q.to(device)
    labels = combined.edge_label.to(device)
    dom_lb = domain_labels.to(device)

    E = combined.edge_label.shape[0]
    all_idx = np.arange(E, dtype=np.int64)

    # Stratified split on the SOURCE-S subset only for early stopping
    src_idx = all_idx[src_mask]
    src_lbl = combined.edge_label[src_mask].numpy()
    src_tr, src_val = _tts(src_idx, test_size=0.2, random_state=0, stratify=src_lbl)

    # Per-domain train indices (over the full combined graph, excluding src_val)
    src_val_set = set(src_val.tolist())
    dom_train = []
    for d in range(3):
        if d == src_domain:
            dom_train.append(src_tr.copy())
        else:
            d_idx = all_idx[(domain_labels == d).numpy()]
            dom_train.append(d_idx.copy())

    best_mcc, best_state, pat_cnt = -2.0, None, 0

    for epoch in range(epochs):
        model.train()
        p     = (epoch + 1) / epochs
        lambd = min(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0, lambda_max)

        for d in range(3):
            np.random.shuffle(dom_train[d])

        min_dom = min(len(dom_train[d]) for d in range(3))
        n_batches = max(min_dom // batch_size, 1)
        ep_cls = ep_dom = 0.0

        for b in range(n_batches):
            s, e = b * batch_size, (b + 1) * batch_size

            # Attack batch: only from src_domain
            atk_ids = dom_train[src_domain][s:e]
            if len(atk_ids) == 0:
                continue
            atk_t = torch.as_tensor(atk_ids, dtype=torch.long, device=device)

            # Domain batch: one block per domain
            dom_np  = np.concatenate([dom_train[d][s:e] for d in range(3)])
            dom_t   = torch.as_tensor(dom_np, dtype=torch.long, device=device)

            optimizer.zero_grad()
            atk_logits, _, _ = model(x, ei[:, atk_t], ea[atk_t], lambd=0.0)
            L_cls = atk_crit(atk_logits, labels[atk_t])
            _, dom_logits, _ = model(x, ei[:, dom_t], ea[dom_t], lambd=lambd)
            L_dom = dom_crit(dom_logits, dom_lb[dom_t])
            loss  = L_cls + L_dom
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_cls += L_cls.item()
            ep_dom += L_dom.item()

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for s2 in range(0, len(src_val), 10000):
                vb = torch.as_tensor(src_val[s2:s2+10000], dtype=torch.long, device=device)
                atk, _, _ = model(x, ei[:, vb], ea[vb], lambd=0.0)
                val_preds.append(atk.argmax(1).cpu().numpy())
                val_trues.append(labels[vb].cpu().numpy())

        val_mcc = compute_mcc(np.concatenate(val_trues), np.concatenate(val_preds))
        log.info(f"  [specialist src={src_domain}] epoch {epoch+1:02d}"
                 f"  L_cls={ep_cls/n_batches:.4f}  L_dom={ep_dom/n_batches:.4f}"
                 f"  val_mcc={val_mcc:.4f}  lam={lambd:.3f}")

        if val_mcc > best_mcc:
            best_mcc  = val_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt   = 0
        else:
            if epoch >= min_epochs:
                pat_cnt += 1
                if pat_cnt >= patience:
                    log.info(f"  Early stop epoch {epoch+1}")
                    break

    log.info(f"  [specialist src={src_domain}] Best val MCC: {best_mcc:.4f}")
    return best_state


def _train_gate(gate, tier_a_graphs, device, epochs=20, batch_size=4096):
    """
    Train gating network to predict source label from quantile-encoded Tier-A features.
    tier_a_graphs: list of 3 graphs with edge_attr_q (Tier-A quantile features).
    """
    all_feat, all_labels = [], []
    for src_idx, g in enumerate(tier_a_graphs):
        all_feat.append(g.edge_attr_q)
        all_labels.append(torch.full((g.edge_attr_q.shape[0],), src_idx, dtype=torch.long))

    X = torch.cat(all_feat, dim=0)     # [E, 4]
    y = torch.cat(all_labels, dim=0)   # [E]

    gate.to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X_dev = X.to(device)
    y_dev = y.to(device)
    E     = X.shape[0]
    idx   = np.arange(E, dtype=np.int64)

    for epoch in range(epochs):
        gate.train()
        np.random.shuffle(idx)
        ep_loss = 0.0
        for s in range(0, E, batch_size):
            b   = torch.as_tensor(idx[s:s+batch_size], dtype=torch.long, device=device)
            out = gate.net(X_dev[b])
            loss = criterion(out, y_dev[b])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            gate.eval()
            with torch.no_grad():
                preds = gate.net(X_dev).argmax(1).cpu().numpy()
            acc = (preds == y.numpy()).mean()
            log.info(f"  Gate epoch {epoch+1}/{epochs}  loss={ep_loss:.4f}  acc={acc:.4f}")

    return {k: v.cpu().clone() for k, v in gate.state_dict().items()}


def _eval_source_gated(specialist_models, gate, test_struct_graph,
                       test_tier_a_graph, device, bs=50000):
    """
    Inference for source-gated model.
    specialist_models: list of 3 DANN_EGS.
    gate: GateNetwork trained on Tier-A features.
    """
    # Gate weights from Tier-A features
    gate.eval().to(device)
    ea_a = test_tier_a_graph.edge_attr_q.to(device)
    E    = ea_a.shape[0]
    gate_weights = []
    with torch.no_grad():
        for s in range(0, E, bs):
            gate_weights.append(gate(ea_a[s:s+bs]).cpu())
    w = torch.cat(gate_weights, dim=0)   # [E, 3]

    # Per-source logits (from structure-only test graph)
    x_t  = test_struct_graph.x.to(device)
    ei_t = test_struct_graph.edge_index.to(device)
    ea_t = test_struct_graph.edge_attr_q.to(device)

    source_logits = []
    for model in specialist_models:
        model.eval().to(device)
        parts = []
        with torch.no_grad():
            for s in range(0, E, bs):
                atk = model.predict(x_t, ei_t[:, s:s+bs], ea_t[s:s+bs])
                parts.append(atk.cpu())
        source_logits.append(torch.cat(parts, dim=0))   # [E, 2]

    stacked = torch.stack(source_logits, dim=1)   # [E, 3, 2]
    # Weighted sum: w[e,s] * logit[e,s,c]
    # w: [E, 3]  → [E, 3, 1]  * stacked [E, 3, 2]
    combined_logits = (w.unsqueeze(-1) * stacked).sum(dim=1)   # [E, 2]
    preds = combined_logits.argmax(dim=1).numpy()
    return preds


# ── E5.0 — DANN multi-seed ───────────────────────────────────────────────────

def run_e5_0_dann_seeds(seeds, lambda_max, dev):
    log.info(f"=== E5.0  DANN seeds={seeds} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for seed in seeds:
        log.info(f"\n  --- Seed {seed} ---")
        for fold in ALL_FOLDS:
            train_dsets = fold["train"]
            test_dset   = fold["test"]
            exp_id      = "E5.0_dann"

            if already_done(exp_id, seed, test_dset):
                log.info(f"  Skipping seed={seed} test={test_dset} (already done)")
                continue

            seed_everything(seed)
            t0 = time.time()
            log.info(f"  Fold: train={train_dsets}  test={test_dset}  seed={seed}")

            combined, test_graph, _, _ = _load_fold_struct(fold, dev)
            val_split     = _make_val_split(combined)
            domain_labels = _get_domain_labels(train_dsets, dev)

            model = DANN_EGS(node_in=8, edge_in=1, hidden=128, num_domains=3)
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
            preds = []
            with torch.no_grad():
                for s in range(0, E_t, 50000):
                    preds.append(model.predict(x_t, ei_t[:, s:s+50000],
                                               ea_t[s:s+50000]).argmax(1).cpu().numpy())
            y_pred  = np.concatenate(preds)
            metrics = compute_all_metrics(test_graph.edge_label.numpy(), y_pred,
                                          y_true_type=test_graph.edge_label_type)
            elapsed = time.time() - t0

            log.info(f"  seed={seed} fold={test_dset}  MCC={metrics['mcc']:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset,
                       "mcc", metrics["mcc"], elapsed)
            log_result(exp_id, seed, train_dsets, test_dset,
                       "macro_f1", metrics["macro_f1"], elapsed)

    _print_multi_seed_summary("E5.0_dann", seeds + [0], "mcc")


def _print_multi_seed_summary(exp_id_new, seeds, metric="mcc"):
    """Print mean ± std across seeds (includes seed 0 from E4.2_dann)."""
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return

    # Collect per-fold, per-seed values
    fold_seed_vals = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["metric"] != metric:
                continue
            eid, sd, td = row["experiment_id"], int(row["seed"]), row["test_dataset"]
            # include both E4.2_dann (seed 0) and E5.0_dann (seeds 1,2)
            if eid in ("E4.2_dann", "E5.0_dann") and sd in seeds:
                fold_seed_vals.setdefault(td, []).append(float(row["value"]))

    log.info(f"\n  Multi-seed summary for DANN ({exp_id_new} + E4.2_dann seed 0):")
    fold_means = []
    for fold_name, vals in sorted(fold_seed_vals.items()):
        m, s = np.mean(vals), np.std(vals)
        fold_means.append(m)
        log.info(f"    {fold_name:<20} mean={m:.4f}  std={s:.4f}  n={len(vals)}")
    if fold_means:
        overall = np.mean(fold_means)
        log.info(f"  Overall mean MCC: {overall:.4f}")
        if overall >= 0.37 - 0.05:
            log.info("  → Result stable. Proceed with E5.2 + E5.3.")
        elif overall >= 0.30:
            log.info("  → Mean dropped but DANN still beats E1.E. Adjust framing.")
        else:
            log.info("  → Fragile. Pivot to mechanism-characterisation paper.")


# ── E5.1 — λ=0 ablation ──────────────────────────────────────────────────────

def run_e5_1_dann_no_grl(seed, dev):
    log.info("=== E5.1  DANN λ=0 ablation (GRL disabled) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold in ALL_FOLDS:
        train_dsets = fold["train"]
        test_dset   = fold["test"]
        exp_id      = "E5.1_dann_no_grl"

        if already_done(exp_id, seed, test_dset):
            log.info(f"  Skipping {exp_id} test={test_dset} (already done)")
            continue

        seed_everything(seed)
        t0 = time.time()
        log.info(f"\n  Fold: train={train_dsets}  test={test_dset}")

        combined, test_graph, _, _ = _load_fold_struct(fold, dev)
        val_split     = _make_val_split(combined)
        domain_labels = _get_domain_labels(train_dsets, dev)

        model = DANN_EGS(node_in=8, edge_in=1, hidden=128, num_domains=3)
        best_state = _train_dann(
            model, combined, domain_labels, val_split, device,
            lambda_max=0.0, epochs=50, patience=7,
        )
        model.load_state_dict(best_state)
        save_model(exp_id, seed, test_dset, best_state)

        model.eval().to(device)
        x_t  = test_graph.x.to(device)
        ei_t = test_graph.edge_index.to(device)
        ea_t = test_graph.edge_attr_q.to(device)
        preds = []
        with torch.no_grad():
            for s in range(0, ei_t.shape[1], 50000):
                preds.append(model.predict(x_t, ei_t[:, s:s+50000],
                                           ea_t[s:s+50000]).argmax(1).cpu().numpy())
        y_pred  = np.concatenate(preds)
        metrics = compute_all_metrics(test_graph.edge_label.numpy(), y_pred,
                                      y_true_type=test_graph.edge_label_type)
        elapsed = time.time() - t0

        log.info(f"  fold={test_dset}  MCC={metrics['mcc']:.4f}")
        log_result(exp_id, seed, train_dsets, test_dset, "mcc", metrics["mcc"], elapsed)
        log_result(exp_id, seed, train_dsets, test_dset, "macro_f1", metrics["macro_f1"], elapsed)

        probe_acc = _run_probe_on_encoder(model.encoder, train_dsets, dev, seed, device)
        log.info(f"  Probe accuracy: {probe_acc:.4f}")
        log_result(exp_id, seed, train_dsets, test_dset, "dataset_probe_acc", probe_acc, 0.0)

    _compare_ablation_summary(seed)


def _compare_ablation_summary(seed):
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return
    dann_mccs, no_grl_mccs = {}, {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["metric"] != "mcc" or row["seed"] != str(seed):
                continue
            td = row["test_dataset"]
            if row["experiment_id"] == "E4.2_dann":
                dann_mccs[td] = float(row["value"])
            elif row["experiment_id"] == "E5.1_dann_no_grl":
                no_grl_mccs[td] = float(row["value"])

    log.info("\n  E5.1 Attribution comparison (seed 0):")
    log.info(f"  {'Fold':<20} {'DANN (λ=1)':<14} {'DANN (λ=0)':<14} {'Δ'}")
    diffs = []
    for td in sorted(set(dann_mccs) | set(no_grl_mccs)):
        v1 = dann_mccs.get(td, float("nan"))
        v0 = no_grl_mccs.get(td, float("nan"))
        d  = v1 - v0 if not (math.isnan(v1) or math.isnan(v0)) else float("nan")
        diffs.append(d)
        log.info(f"  {td:<20} {v1:<14.4f} {v0:<14.4f} {d:+.4f}")

    valid = [d for d in diffs if not math.isnan(d)]
    if valid:
        mean_diff = np.mean(valid)
        log.info(f"  Mean DANN advantage from GRL: {mean_diff:+.4f}")
        if mean_diff >= 0.05:
            log.info("  → Adversarial mechanism is real (≥+0.05). Strong paper evidence.")
        elif mean_diff >= 0.0:
            log.info("  → Auxiliary-head explains part of gain. Reframe as multi-task + adversarial.")
        else:
            log.info("  → GRL hurts or does nothing. Gain is purely from auxiliary task head.")


# ── E5.2 — Source-Conditional Gating ─────────────────────────────────────────

def run_e5_2_source_gated(seed, dev, lambda_max=1.0):
    log.info("=== E5.2  Source-Conditional Gating ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold in ALL_FOLDS:
        train_dsets = fold["train"]
        test_dset   = fold["test"]
        exp_id      = "E5.2_source_gated"

        if already_done(exp_id, seed, test_dset):
            log.info(f"  Skipping {exp_id} test={test_dset} (already done)")
            continue

        seed_everything(seed)
        t0 = time.time()
        log.info(f"\n  Fold: train={train_dsets}  test={test_dset}")

        # Structure-only combined graph + domain labels
        combined, test_struct, _, _ = _load_fold_struct(fold, dev)
        domain_labels = _get_domain_labels(train_dsets, dev)

        # Tier-A graphs for gate training and inference
        tier_a_train = [_build_tier_a_graph(ds, dev) for ds in train_dsets]
        tier_a_test  = _build_tier_a_graph(test_dset, dev)

        # ── Train 3 per-source specialist DANN models ──────────────────────
        specialists = []
        for src_d, ds in enumerate(train_dsets):
            ckpt = Path(f"results/models/E5.2_spec{src_d}_seed{seed}_test{test_dset}.pt")
            model = DANN_EGS(node_in=8, edge_in=1, hidden=128, num_domains=3)
            if ckpt.exists():
                log.info(f"  Loading specialist {src_d} from {ckpt}")
                model.load_state_dict(torch.load(ckpt, weights_only=True))
                model.to(device)
            else:
                log.info(f"  Training specialist for source {src_d} ({ds})…")
                best_state = _train_source_specialist(
                    model, combined, domain_labels, src_domain=src_d,
                    device=device, lambda_max=lambda_max,
                    epochs=50, patience=7,
                )
                torch.save(best_state, ckpt)
                model.load_state_dict(best_state)
                model.to(device)
            specialists.append(model)

        # ── Train gating network on Tier-A features ────────────────────────
        gate_ckpt = Path(f"results/models/E5.2_gate_seed{seed}_test{test_dset}.pt")
        gate = GateNetwork(in_features=tier_a_train[0].edge_attr_q.shape[1],
                           hidden=64, num_sources=3)
        if gate_ckpt.exists():
            log.info(f"  Loading gate from {gate_ckpt}")
            gate.load_state_dict(torch.load(gate_ckpt, weights_only=True))
        else:
            log.info("  Training gate…")
            gate_state = _train_gate(gate, tier_a_train, device)
            torch.save(gate_state, gate_ckpt)
            gate.load_state_dict(gate_state)

        # Sanity check: gate calibration on training data
        gate.eval().to(device)
        with torch.no_grad():
            gate_preds = []
            for g in tier_a_train:
                ea_q = g.edge_attr_q.to(device)
                gate_preds.append(gate(ea_q).argmax(1).cpu().numpy())
        gate_preds_all = np.concatenate(gate_preds)
        gate_true_all  = np.concatenate([np.full(g.edge_attr_q.shape[0], i)
                                          for i, g in enumerate(tier_a_train)])
        gate_acc = (gate_preds_all == gate_true_all).mean()
        log.info(f"  Gate train accuracy: {gate_acc:.4f}")

        # ── Inference ─────────────────────────────────────────────────────
        y_pred = _eval_source_gated(specialists, gate, test_struct,
                                     tier_a_test, device)
        y_true  = test_struct.edge_label.numpy()
        metrics = compute_all_metrics(y_true, y_pred,
                                      y_true_type=test_struct.edge_label_type)
        elapsed = time.time() - t0

        ref  = E1E_REF.get(test_dset)
        log.info(f"  fold={test_dset}  MCC={metrics['mcc']:.4f}"
                 f"  macro_F1={metrics['macro_f1']:.4f}"
                 f"  (E1.E_ref={ref})")
        log_result(exp_id, seed, train_dsets, test_dset,
                   "mcc", metrics["mcc"], elapsed)
        log_result(exp_id, seed, train_dsets, test_dset,
                   "macro_f1", metrics["macro_f1"], elapsed)
        log_result(exp_id, seed, train_dsets, test_dset,
                   "gate_train_acc", gate_acc, 0.0)

    log.info("\n  E5.2 Decision summary:")
    _print_fold_summary(exp_id, seed, metric="mcc")


def _build_tier_a_graph(ds, dev):
    """Load Tier-A graph (4 features: byte_count, pkt_count, tcp_flags, duration_ms)."""
    g = load_graph(ds, tier="A", dev=dev)
    # Quantile-encode Tier-A features per-graph
    ea_q = quantile_encode(g.edge_attr)
    g2   = copy.copy(g)
    g2.edge_attr_q = ea_q
    return g2


# ── E5.3 — CDAN ──────────────────────────────────────────────────────────────

def _train_cdan(model, combined, domain_labels, val_split,
                device, lambda_max, epochs=50, patience=7, min_epochs=5,
                batch_size_per_domain=1024):
    """Same loop structure as _train_dann but uses CDAN_EGS forward pass."""
    model.to(device)
    optimizer        = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    cw               = _class_weights(combined.edge_label, device=device)
    attack_criterion = nn.CrossEntropyLoss(weight=cw)
    domain_criterion = nn.CrossEntropyLoss()

    x      = combined.x.to(device)
    ei     = combined.edge_index.to(device)
    ea     = combined.edge_attr_q.to(device)
    labels = combined.edge_label.to(device)
    dom_lb = domain_labels.to(device)

    train_idx = np.array(val_split["train"], dtype=np.int64)
    val_idx   = val_split["val"]

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
        ep_cls = ep_dom = 0.0

        for b in range(n_batches):
            s, e  = b * batch_size_per_domain, (b + 1) * batch_size_per_domain
            batch = np.concatenate([dom_train[d][s:e] for d in range(3)])
            bt    = torch.as_tensor(batch, dtype=torch.long, device=device)

            optimizer.zero_grad()
            atk_logits, dom_logits, _ = model(x, ei[:, bt], ea[bt], lambd=lambd)
            L_cls = attack_criterion(atk_logits, labels[bt])
            L_dom = domain_criterion(dom_logits, dom_lb[bt])
            (L_cls + L_dom).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_cls += L_cls.item()
            ep_dom += L_dom.item()

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for s in range(0, len(val_idx), 10000):
                vb = torch.as_tensor(val_idx[s:s+10000], dtype=torch.long, device=device)
                atk, _, _ = model(x, ei[:, vb], ea[vb], lambd=0.0)
                val_preds.append(atk.argmax(1).cpu().numpy())
                val_trues.append(labels[vb].cpu().numpy())

        val_mcc = compute_mcc(np.concatenate(val_trues), np.concatenate(val_preds))
        log.info(f"  CDAN epoch {epoch+1:02d}  L_cls={ep_cls/max(n_batches,1):.4f}"
                 f"  L_dom={ep_dom/max(n_batches,1):.4f}  val_mcc={val_mcc:.4f}"
                 f"  lam={lambd:.3f}")

        if val_mcc > best_mcc:
            best_mcc  = val_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt   = 0
        else:
            if epoch >= min_epochs:
                pat_cnt += 1
                if pat_cnt >= patience:
                    log.info(f"  Early stop epoch {epoch+1}")
                    break

    log.info(f"  Best CDAN val MCC: {best_mcc:.4f}")
    return best_state


def run_e5_3_cdan(seed, dev, lambda_max=1.0):
    log.info(f"=== E5.3  CDAN (class-conditional DANN)  lambda_max={lambda_max} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold in ALL_FOLDS:
        train_dsets = fold["train"]
        test_dset   = fold["test"]
        exp_id      = "E5.3_cdan"

        if already_done(exp_id, seed, test_dset):
            log.info(f"  Skipping {exp_id} test={test_dset} (already done)")
            continue

        seed_everything(seed)
        t0 = time.time()
        log.info(f"\n  Fold: train={train_dsets}  test={test_dset}")

        combined, test_graph, _, _ = _load_fold_struct(fold, dev)
        val_split     = _make_val_split(combined)
        domain_labels = _get_domain_labels(train_dsets, dev)

        model = CDAN_EGS(node_in=8, edge_in=1, hidden=128, num_domains=3)
        best_state = _train_cdan(
            model, combined, domain_labels, val_split, device,
            lambda_max=lambda_max, epochs=50, patience=7,
        )
        model.load_state_dict(best_state)
        save_model(exp_id, seed, test_dset, best_state)

        model.eval().to(device)
        x_t  = test_graph.x.to(device)
        ei_t = test_graph.edge_index.to(device)
        ea_t = test_graph.edge_attr_q.to(device)
        preds = []
        with torch.no_grad():
            for s in range(0, ei_t.shape[1], 50000):
                preds.append(model.predict(x_t, ei_t[:, s:s+50000],
                                           ea_t[s:s+50000]).argmax(1).cpu().numpy())
        y_pred  = np.concatenate(preds)
        metrics = compute_all_metrics(test_graph.edge_label.numpy(), y_pred,
                                      y_true_type=test_graph.edge_label_type)
        elapsed = time.time() - t0

        log.info(f"  fold={test_dset}  MCC={metrics['mcc']:.4f}"
                 f"  macro_F1={metrics['macro_f1']:.4f}")
        log_result(exp_id, seed, train_dsets, test_dset,
                   "mcc", metrics["mcc"], elapsed)
        log_result(exp_id, seed, train_dsets, test_dset,
                   "macro_f1", metrics["macro_f1"], elapsed)

        probe_acc = _run_probe_on_encoder(model.encoder, train_dsets, dev, seed, device)
        log.info(f"  Probe accuracy: {probe_acc:.4f}")
        log_result(exp_id, seed, train_dsets, test_dset,
                   "dataset_probe_acc", probe_acc, 0.0)

    log.info("\n  E5.3 vs E4.2 comparison:")
    _compare_two_exps("E4.2_dann", "E5.3_cdan", seed)


def _compare_two_exps(exp_a, exp_b, seed):
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return
    a_vals, b_vals = {}, {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["metric"] != "mcc" or row["seed"] != str(seed):
                continue
            if row["experiment_id"] == exp_a:
                a_vals[row["test_dataset"]] = float(row["value"])
            elif row["experiment_id"] == exp_b:
                b_vals[row["test_dataset"]] = float(row["value"])
    log.info(f"  {'Fold':<20} {exp_a:<18} {exp_b:<18} {'Δ'}")
    diffs = []
    for td in sorted(set(a_vals) | set(b_vals)):
        va = a_vals.get(td, float("nan"))
        vb = b_vals.get(td, float("nan"))
        d  = vb - va if not (math.isnan(va) or math.isnan(vb)) else float("nan")
        diffs.append(d)
        log.info(f"  {td:<20} {va:<18.4f} {vb:<18.4f} {d:+.4f}")
    valid = [d for d in diffs if not math.isnan(d)]
    if valid:
        mean_diff = np.mean(valid)
        log.info(f"  Mean {exp_b} vs {exp_a}: {mean_diff:+.4f}")
        if mean_diff >= 0.05:
            log.info(f"  → {exp_b} wins. Adopt as primary method.")
        elif mean_diff >= -0.05:
            log.info(f"  → Roughly equivalent. Report as confirmed alternative.")
        else:
            log.info(f"  → {exp_b} underperforms. Marginal alignment was right level; keep {exp_a}.")


# ── E5.4 — DropEdge augmentation ─────────────────────────────────────────────

def _determine_best_exp(seed):
    """Scan results.csv and return the exp_id with highest mean MCC across 4 folds."""
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return "dann"
    fold_vals = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["metric"] != "mcc" or row["seed"] != str(seed):
                continue
            eid = row["experiment_id"]
            fold_vals.setdefault(eid, []).append(float(row["value"]))
    if not fold_vals:
        return "dann"
    best_eid = max(fold_vals, key=lambda k: np.mean(fold_vals[k]))
    log.info(f"  Best exp from results.csv: {best_eid}"
             f"  mean_mcc={np.mean(fold_vals[best_eid]):.4f}")
    # Translate experiment_id to architecture key
    if "cdan" in best_eid:
        return "cdan"
    if "source_gated" in best_eid:
        return "source_gated"
    return "dann"


def run_e5_4_dropedge(p_drops, base, seed, dev, lambda_max=1.0):
    log.info(f"=== E5.4  DropEdge  p_drops={p_drops}  base={base} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if base == "best":
        base = _determine_best_exp(seed)
        log.info(f"  Resolved --base best → {base}")

    from torch_geometric.utils import dropout_edge as pyg_dropout_edge

    for p_drop in p_drops:
        for fold in ALL_FOLDS:
            train_dsets = fold["train"]
            test_dset   = fold["test"]
            exp_id      = f"E5.4_dropedge_p{int(p_drop*100):02d}_{base}"

            if already_done(exp_id, seed, test_dset):
                log.info(f"  Skipping {exp_id} test={test_dset} (already done)")
                continue

            seed_everything(seed)
            t0 = time.time()
            log.info(f"\n  Fold: train={train_dsets}  test={test_dset}  p_drop={p_drop}")

            combined, test_graph, _, _ = _load_fold_struct(fold, dev)
            val_split     = _make_val_split(combined)
            domain_labels = _get_domain_labels(train_dsets, dev)

            if base == "cdan":
                model = CDAN_EGS(node_in=8, edge_in=1, hidden=128, num_domains=3)
                best_state = _train_dann_dropedge(
                    model, combined, domain_labels, val_split, device,
                    lambda_max=lambda_max, p_drop=p_drop,
                    is_cdan=True,
                )
            else:  # dann (default)
                model = DANN_EGS(node_in=8, edge_in=1, hidden=128, num_domains=3)
                best_state = _train_dann_dropedge(
                    model, combined, domain_labels, val_split, device,
                    lambda_max=lambda_max, p_drop=p_drop,
                    is_cdan=False,
                )

            model.load_state_dict(best_state)
            save_model(exp_id, seed, test_dset, best_state)

            model.eval().to(device)
            x_t  = test_graph.x.to(device)
            ei_t = test_graph.edge_index.to(device)
            ea_t = test_graph.edge_attr_q.to(device)
            preds = []
            with torch.no_grad():
                for s in range(0, ei_t.shape[1], 50000):
                    preds.append(model.predict(x_t, ei_t[:, s:s+50000],
                                               ea_t[s:s+50000]).argmax(1).cpu().numpy())
            y_pred  = np.concatenate(preds)
            metrics = compute_all_metrics(test_graph.edge_label.numpy(), y_pred,
                                          y_true_type=test_graph.edge_label_type)
            elapsed = time.time() - t0

            log.info(f"  p_drop={p_drop}  fold={test_dset}  MCC={metrics['mcc']:.4f}"
                     f"  macro_F1={metrics['macro_f1']:.4f}")
            log_result(exp_id, seed, train_dsets, test_dset,
                       "mcc", metrics["mcc"], elapsed)
            log_result(exp_id, seed, train_dsets, test_dset,
                       "macro_f1", metrics["macro_f1"], elapsed)

    # Summary
    for p_drop in p_drops:
        eid = f"E5.4_dropedge_p{int(p_drop*100):02d}_{base}"
        log.info(f"\n  DropEdge p={p_drop} summary:")
        _print_fold_summary(eid, seed, metric="mcc")


def _train_dann_dropedge(model, combined, domain_labels, val_split,
                          device, lambda_max, p_drop,
                          epochs=50, patience=7, min_epochs=5,
                          batch_size_per_domain=1024,
                          is_cdan=False):
    """
    DANN/CDAN training with epoch-level DropEdge:
    each epoch randomly drops p_drop fraction of training edges before mini-batching.
    """
    from torch_geometric.utils import dropout_edge as pyg_dropout_edge

    model.to(device)
    optimizer        = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    cw               = _class_weights(combined.edge_label, device=device)
    attack_criterion = nn.CrossEntropyLoss(weight=cw)
    domain_criterion = nn.CrossEntropyLoss()

    x_cpu  = combined.x
    ei_cpu = combined.edge_index
    ea_cpu = combined.edge_attr_q
    labels = combined.edge_label
    dom_lb = domain_labels

    train_idx = np.array(val_split["train"], dtype=np.int64)
    val_idx   = val_split["val"]

    dom_train = []
    for d in range(3):
        mask = (domain_labels[train_idx] == d).numpy()
        dom_train.append(train_idx[mask].copy())

    best_mcc, best_state, pat_cnt = -2.0, None, 0

    for epoch in range(epochs):
        model.train()
        p     = (epoch + 1) / epochs
        lambd = min(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0, lambda_max)

        # DropEdge: subsample training indices for this epoch
        epoch_dom = []
        for d in range(3):
            n_keep = max(int(len(dom_train[d]) * (1.0 - p_drop)), 1)
            keep   = np.random.choice(dom_train[d], size=n_keep, replace=False)
            epoch_dom.append(keep)

        # Re-sort kept indices to preserve temporal order
        x      = x_cpu.to(device)
        ei     = ei_cpu.to(device)
        ea     = ea_cpu.to(device)
        lbl    = labels.to(device)
        dom_l  = dom_lb.to(device)

        for d in range(3):
            np.random.shuffle(epoch_dom[d])

        n_batches = min(len(d) for d in epoch_dom) // batch_size_per_domain
        ep_cls = ep_dom = 0.0

        for b in range(n_batches):
            s, e  = b * batch_size_per_domain, (b + 1) * batch_size_per_domain
            batch = np.concatenate([epoch_dom[d][s:e] for d in range(3)])
            bt    = torch.as_tensor(batch, dtype=torch.long, device=device)

            optimizer.zero_grad()
            if is_cdan:
                atk_logits, dom_logits, _ = model(x, ei[:, bt], ea[bt], lambd=lambd)
            else:
                atk_logits, dom_logits, _ = model(x, ei[:, bt], ea[bt], lambd=lambd)
            L_cls = attack_criterion(atk_logits, lbl[bt])
            L_dom = domain_criterion(dom_logits, dom_l[bt])
            (L_cls + L_dom).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_cls += L_cls.item()
            ep_dom += L_dom.item()

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for s in range(0, len(val_idx), 10000):
                vb = torch.as_tensor(val_idx[s:s+10000], dtype=torch.long, device=device)
                atk, _, _ = model(x, ei[:, vb], ea[vb], lambd=0.0)
                val_preds.append(atk.argmax(1).cpu().numpy())
                val_trues.append(lbl[vb].cpu().numpy())

        val_mcc = compute_mcc(np.concatenate(val_trues), np.concatenate(val_preds))
        log.info(f"  DropEdge(p={p_drop}) epoch {epoch+1:02d}"
                 f"  L_cls={ep_cls/max(n_batches,1):.4f}"
                 f"  L_dom={ep_dom/max(n_batches,1):.4f}"
                 f"  val_mcc={val_mcc:.4f}  lam={lambd:.3f}")

        if val_mcc > best_mcc:
            best_mcc  = val_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt   = 0
        else:
            if epoch >= min_epochs:
                pat_cnt += 1
                if pat_cnt >= patience:
                    log.info(f"  Early stop epoch {epoch+1}")
                    break

    log.info(f"  Best DropEdge val MCC: {best_mcc:.4f}")
    return best_state


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Phase 5 experiments (spex5.md)")
    parser.add_argument("--exp", required=True,
                        choices=["dann", "dann_no_grl", "source_gated",
                                 "cdan", "dropedge", "all"])
    parser.add_argument("--seeds",     nargs="+", type=int, default=[1, 2],
                        help="Seeds for E5.0 (--exp dann).  E.g. --seeds 1 2")
    parser.add_argument("--seed",      type=int, default=0,
                        help="Single seed for all other experiments")
    parser.add_argument("--lambda_max", type=float, default=1.0)
    parser.add_argument("--p_drop",    nargs="+", type=float, default=[0.2, 0.5],
                        help="DropEdge fractions (--exp dropedge)")
    parser.add_argument("--base",      default="dann",
                        help="Base architecture for DropEdge: dann | cdan | source_gated | best")
    parser.add_argument("--dev",       action="store_true", default=True)
    parser.add_argument("--no-dev",    dest="dev", action="store_false")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.exp in ("dann", "all"):
        run_e5_0_dann_seeds(args.seeds, args.lambda_max, args.dev)

    if args.exp in ("dann_no_grl", "all"):
        run_e5_1_dann_no_grl(args.seed, args.dev)

    if args.exp in ("source_gated", "all"):
        run_e5_2_source_gated(args.seed, args.dev, args.lambda_max)

    if args.exp in ("cdan", "all"):
        run_e5_3_cdan(args.seed, args.dev, args.lambda_max)

    if args.exp in ("dropedge", "all"):
        run_e5_4_dropedge(args.p_drop, args.base, args.seed, args.dev, args.lambda_max)


if __name__ == "__main__":
    main()
