#!/usr/bin/env python3
"""
Phase 3 follow-up experiments (spex3.md), excluding TGN (P3.1).
All 4 LODO folds, seed 0. Reuses Phase 1/2 code paths.

Usage:
    python scripts/run_phase3.py --exp per_attack      [--seed 0]
    python scripts/run_phase3.py --exp cic17_single    [--seed 0]
    python scripts/run_phase3.py --exp embed_analysis  [--models cic18 ton] [--seed 0]
    python scripts/run_phase3.py --exp scale_invariant [--folds all] [--seed 0]
    python scripts/run_phase3.py --exp all             [--seed 0]

    Add --no-dev to use full datasets (default: dev subsamples).
"""

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.model_selection import train_test_split as _tts

from src.utils.logging import setup_logging, log_result, already_done, save_model
from src.utils.seeding import seed_everything
from src.utils.metrics import compute_all_metrics, CLASSES
from src.data.graph_builder import load_graph, combine_graphs
from src.models.egraphsage import EdgeAwareSAGE
from src.train.train_loops import train_egraphsage
from src.train.eval import eval_egraphsage

log = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

ALL_FOLDS = [
    {"train": ["cic_ids2018",   "unsw_nb15",   "ton_iot"],       "test": "lycos_ids2017"},
    {"train": ["lycos_ids2017", "unsw_nb15",   "ton_iot"],       "test": "cic_ids2018"},
    {"train": ["lycos_ids2017", "cic_ids2018", "ton_iot"],       "test": "unsw_nb15"},
    {"train": ["lycos_ids2017", "cic_ids2018", "unsw_nb15"],     "test": "ton_iot"},
]

NAME_MAP = {
    "cic18": "cic_ids2018", "ton": "ton_iot",
    "lycos": "lycos_ids2017", "unsw": "unsw_nb15",
}

FIGURES_DIR = Path("results/figures/phase3")


# ── data helpers ─────────────────────────────────────────────────────────────

def _load_fold(fold, dev, structure_only=False, tier="B"):
    train_dsets = fold["train"]
    test_dset   = fold["test"]

    train_graphs = [load_graph(ds, tier=tier, dev=dev) for ds in train_dsets]
    combined     = combine_graphs(train_graphs)
    test_graph   = load_graph(test_dset, tier=tier, dev=dev)

    max_feat = combined.edge_attr.shape[1]
    d = test_graph.edge_attr.shape[1]
    if d < max_feat:
        pad = torch.zeros(test_graph.edge_attr.shape[0], max_feat - d)
        test_graph.edge_attr   = torch.cat([test_graph.edge_attr, pad], dim=1)
        test_graph.edge_attr_q = torch.cat([test_graph.edge_attr_q, pad], dim=1)
    elif d > max_feat:
        test_graph.edge_attr   = test_graph.edge_attr[:, :max_feat]
        test_graph.edge_attr_q = test_graph.edge_attr_q[:, :max_feat]

    if structure_only:
        combined.edge_attr     = torch.ones_like(combined.edge_attr)
        combined.edge_attr_q   = torch.ones_like(combined.edge_attr_q)
        test_graph.edge_attr   = torch.ones_like(test_graph.edge_attr)
        test_graph.edge_attr_q = torch.ones_like(test_graph.edge_attr_q)

    return combined, test_graph, train_dsets, test_dset


def _make_val_split(combined):
    n = combined.edge_label.shape[0]
    ti, vi = _tts(np.arange(n), test_size=0.2, random_state=0,
                  stratify=combined.edge_label.numpy())
    return {"train": ti.tolist(), "val": vi.tolist()}


def _train_e1e(combined, device, seed=0):
    model = EdgeAwareSAGE(node_in=combined.x.shape[1],
                          edge_in=combined.edge_attr.shape[1])
    val_split = _make_val_split(combined)
    best_state = train_egraphsage(model, combined, val_split=val_split,
                                  device=device, use_quantile=True)
    model.load_state_dict(best_state)
    return model, best_state


def _load_or_train_e1e(combined, test_dset, device, seed=0, exp_prefix="P3.E1E"):
    """Try phase-2 or phase-3 checkpoints; train fresh if none found."""
    for eid in ("P2.Q1b", "P2.Q4a", exp_prefix):
        p = Path(f"results/models/{eid}_seed{seed}_test{test_dset}.pt")
        if p.exists():
            log.info(f"  Loading {p}")
            model = EdgeAwareSAGE(node_in=combined.x.shape[1],
                                  edge_in=combined.edge_attr.shape[1])
            model.load_state_dict(torch.load(p, weights_only=True))
            return model
    log.info(f"  No checkpoint found — training E1.E (structure-only) for test={test_dset}")
    seed_everything(seed)
    model, state = _train_e1e(combined, device, seed)
    save_model(exp_prefix, seed, test_dset, state)
    return model


def _extract_embs(model, graph, device, max_feat, structure_only=True, bs=50000):
    """Return [E, 3H] pre-classifier embeddings for all edges in graph."""
    model.eval().to(device)
    g_x  = graph.x.to(device)
    g_ei = graph.edge_index.to(device)
    if structure_only:
        g_ea = torch.ones(graph.edge_attr.shape[0], max_feat).to(device)
    else:
        g_ea = graph.edge_attr_q.to(device)
    embs = []
    with torch.no_grad():
        for start in range(0, g_ei.shape[1], bs):
            end = min(start + bs, g_ei.shape[1])
            embs.append(model.embed(g_x, g_ei[:, start:end], g_ea[start:end]).cpu().numpy())
    return np.concatenate(embs, axis=0)


def _build_scale_inv_graph(tier_a_graph):
    """
    Compute 11 scale-invariant features from Tier-A raw features.
    Tier-A layout: [byte_count, packet_count, tcp_flags_any, flow_duration_ms]
    Returns a copy of the graph with edge_attr/edge_attr_q replaced.
    """
    from src.data.graph_builder import quantile_encode

    ea = tier_a_graph.edge_attr  # [E, 4]
    byte_count   = ea[:, 0]
    pkt_count    = ea[:, 1]
    tcp_flags    = ea[:, 2].long().clamp(0, 255)  # treat as integer bitmask
    dur_ms       = ea[:, 3]

    bytes_per_pkt  = byte_count / pkt_count.clamp(min=1)
    pkts_per_sec   = pkt_count  / (dur_ms.clamp(min=1) / 1000.0)
    popcount       = sum((tcp_flags >> i) & 1 for i in range(8)).float()
    flag_density   = popcount / pkt_count.clamp(min=1)
    tcp_flag_bits  = torch.stack(
        [((tcp_flags >> i) & 1).float() for i in range(8)], dim=1
    )  # [E, 8]

    new_ea = torch.stack([bytes_per_pkt, pkts_per_sec, flag_density], dim=1)
    new_ea = torch.cat([new_ea, tcp_flag_bits], dim=1)            # [E, 11]
    new_ea = torch.nan_to_num(new_ea, nan=0.0, posinf=0.0, neginf=0.0)

    new_ea_q = quantile_encode(new_ea)

    g = copy.copy(tier_a_graph)
    g.edge_attr   = new_ea
    g.edge_attr_q = new_ea_q
    return g


# ── P3.2 — Per-attack F1 for E1.E (all 4 folds) ─────────────────────────────

def run_p3_2_per_attack(seed, dev):
    log.info("=== P3.2  Per-attack F1 for E1.E (all 4 folds) ===")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    attack_classes = CLASSES[1:]  # skip Benign
    results_table  = {}           # {test_dset: {cls: f1}}

    for fold in ALL_FOLDS:
        combined, test_graph, train_dsets, test_dset = _load_fold(
            fold, dev, structure_only=True)

        model = _load_or_train_e1e(combined, test_dset, device, seed,
                                   exp_prefix="P3.E1E")
        result  = eval_egraphsage(model, test_graph, device=device,
                                  use_quantile=True)
        metrics = compute_all_metrics(result["y_true"], result["y_pred"],
                                      y_true_type=test_graph.edge_label_type)

        if not already_done("P3.2_perattack", seed, test_dset):
            log_result("P3.2_perattack", seed, train_dsets, test_dset,
                       "mcc", metrics["mcc"], 0.0)
            for cls, f1 in metrics.get("per_class_f1", {}).items():
                log_result("P3.2_perattack", seed, train_dsets, test_dset,
                           f"f1_{cls}", f1, 0.0)

        results_table[test_dset] = metrics.get("per_class_f1", {})
        log.info(f"  fold={test_dset}  MCC={metrics['mcc']:.4f}")

    # Print 6×4 table
    folds_order = [f["test"] for f in ALL_FOLDS]
    header = "Attack class             " + "  ".join(f"{d:<18}" for d in folds_order)
    log.info(f"\n  Per-attack F1 table:\n  {header}")
    for cls in attack_classes:
        row = f"  {cls:<24}" + "  ".join(
            f"{results_table.get(d, {}).get(cls, float('nan')):.4f}            "
            for d in folds_order
        )
        log.info(row)

    # Bar plot: mean F1 per attack class, sorted
    mean_f1 = {}
    for cls in attack_classes:
        vals = [results_table.get(d, {}).get(cls, float("nan")) for d in folds_order]
        mean_f1[cls] = float(np.nanmean(vals))

    sorted_cls = sorted(mean_f1, key=mean_f1.get, reverse=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(sorted_cls)), [mean_f1[c] for c in sorted_cls], color="steelblue")
    ax.set_xticks(range(len(sorted_cls)))
    ax.set_xticklabels(sorted_cls, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean F1 across folds")
    ax.set_ylim(0, 1)
    ax.set_title("P3.2  E1.E per-attack mean F1 (structure-only, 4 LODO folds)")
    fig.tight_layout()
    out = FIGURES_DIR / "p3_2_per_attack_f1.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    log.info(f"\n  Saved {out}")

    log.info("\n  Decision:")
    topology_cls = {"Reconnaissance", "DoS_DDoS", "Botnet_C2"}
    feature_cls  = {"Injection_Exploit", "BruteForce"}
    topo_mean  = np.nanmean([mean_f1.get(c, np.nan) for c in topology_cls])
    feat_mean  = np.nanmean([mean_f1.get(c, np.nan) for c in feature_cls])
    log.info(f"    Topology-driven (Recon/DDoS/Botnet) mean F1 = {topo_mean:.4f}")
    log.info(f"    Feature-driven  (Injection/BruteForce) mean F1 = {feat_mean:.4f}")
    if topo_mean > feat_mean + 0.05:
        log.info("    → Mechanism story supported: structural attacks transfer better")
    elif feat_mean > topo_mean + 0.05:
        log.info("    → Inverted: mechanism story wrong, needs reframing")
    else:
        log.info("    → Uniform: structural transfer more general than hypothesis")


# ── P3.3 — CIC17 single-source diagnostics ───────────────────────────────────

def run_p3_3_cic17_single(seed, dev):
    log.info("=== P3.3  CIC17 single-source diagnostics ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dset = "lycos_ids2017"
    test_graph = load_graph(test_dset, tier="B", dev=dev)

    single_sources = ["cic_ids2018", "unsw_nb15", "ton_iot"]

    configs = [(ds,) for ds in single_sources]
    configs.append(tuple(single_sources))  # (d) all-three

    config_labels = {
        ("cic_ids2018",):                      "(a) CIC18 only",
        ("unsw_nb15",):                         "(b) UNSW only",
        ("ton_iot",):                           "(c) ToN only",
        ("cic_ids2018", "unsw_nb15", "ton_iot"): "(d) all-three",
    }

    results = {}
    for cfg in configs:
        exp_id   = f"P3.3_{'+'.join(cfg)}"
        label    = config_labels[cfg]
        log.info(f"\n  Training config {label}: train={list(cfg)}, test={test_dset}")

        if already_done(exp_id, seed, test_dset):
            log.info(f"  Skipping {exp_id} (already done)")
            continue

        seed_everything(seed)
        t0 = time.time()

        if len(cfg) == 1:
            # Single-source: load one dataset, align feature dim with test
            train_g = load_graph(cfg[0], tier="B", dev=dev)
            max_feat = train_g.edge_attr.shape[1]
            # Always deep-copy so ones_like doesn't corrupt test_graph for next iteration
            test_g = copy.deepcopy(test_graph)
            d = test_g.edge_attr.shape[1]
            if d < max_feat:
                pad = torch.zeros(test_g.edge_attr.shape[0], max_feat - d)
                test_g.edge_attr   = torch.cat([test_g.edge_attr, pad], dim=1)
                test_g.edge_attr_q = torch.cat([test_g.edge_attr_q, pad], dim=1)
            elif d > max_feat:
                test_g.edge_attr   = test_g.edge_attr[:, :max_feat]
                test_g.edge_attr_q = test_g.edge_attr_q[:, :max_feat]
            train_g.edge_attr   = torch.ones_like(train_g.edge_attr)
            train_g.edge_attr_q = torch.ones_like(train_g.edge_attr_q)
            test_g.edge_attr    = torch.ones_like(test_g.edge_attr)
            test_g.edge_attr_q  = torch.ones_like(test_g.edge_attr_q)
        else:
            # All-three: standard LODO fold for lycos
            fold = next(f for f in ALL_FOLDS if f["test"] == test_dset)
            train_g, test_g, _, _ = _load_fold(fold, dev, structure_only=True)

        model, _ = _train_e1e(train_g, device, seed)
        result   = eval_egraphsage(model, test_g, device=device, use_quantile=True)
        metrics  = compute_all_metrics(result["y_true"], result["y_pred"],
                                       y_true_type=test_g.edge_label_type)
        elapsed  = time.time() - t0

        log_result(exp_id, seed, list(cfg), test_dset, "mcc", metrics["mcc"], elapsed)
        results[label] = metrics["mcc"]
        log.info(f"  {label}: MCC={metrics['mcc']:.4f}  elapsed={elapsed/60:.1f}min")

    log.info("\n  P3.3 Decision summary:")
    for lbl, mcc in results.items():
        log.info(f"    {lbl}: MCC={mcc:.4f}")
    if results:
        a_mcc = results.get("(a) CIC18 only", float("nan"))
        d_mcc = results.get("(d) all-three",  float("nan"))
        if not np.isnan(a_mcc) and not np.isnan(d_mcc):
            if a_mcc > d_mcc + 0.05:
                log.info("    → UNSW/ToN poisoning CIC17 transfer: single-source (a) >> all-three (d)")
            elif all(v < 0.1 for v in results.values() if not np.isnan(v)):
                log.info("    → CIC17 genuinely hard; all configs near zero (attack-type mismatch)")
            else:
                log.info("    → Mixed result — see per-config MCC above")


# ── P3.4 — E1.E embedding analysis ───────────────────────────────────────────

def run_p3_4_embed_analysis(model_folds, seed, dev):
    """
    model_folds: list of short fold names, e.g. ["cic18", "ton"]
    P3.4a: linear probe on dataset identity (3-way from training embeddings)
    P3.4b: UMAP on 20k-per-dataset training embeddings, colored by attack type + source dataset
    """
    log.info("=== P3.4  E1.E embedding analysis ===")
    try:
        import umap as umap_mod
    except ImportError:
        log.error("  umap-learn not installed.  pip install umap-learn  then re-run.")
        return

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    target_tests = [NAME_MAP.get(m, m) for m in model_folds]
    folds_to_run = [f for f in ALL_FOLDS if f["test"] in target_tests]

    for fold in folds_to_run:
        combined, _, train_dsets, test_dset = _load_fold(fold, dev, structure_only=True)
        max_feat = combined.edge_attr.shape[1]
        seed_everything(seed)

        model = _load_or_train_e1e(combined, test_dset, device, seed)

        # ── P3.4a — Linear probe ──────────────────────────────────────────────
        log.info(f"\n  P3.4a  Linear probe: fold={test_dset}, train_dsets={train_dsets}")

        MAX_PER_DSET = 10000
        all_embs, all_labels = [], []
        for ds_idx, ds in enumerate(train_dsets):
            g    = load_graph(ds, tier="B", dev=dev)
            embs = _extract_embs(model, g, device, max_feat)
            rng  = np.random.RandomState(seed)
            idx  = rng.choice(len(embs), min(MAX_PER_DSET, len(embs)), replace=False)
            all_embs.append(embs[idx])
            all_labels.extend([ds_idx] * len(idx))
            log.info(f"    {ds}: {len(idx)} training embeddings")

        X = np.concatenate(all_embs)
        y = np.array(all_labels)

        ti, vi = _tts(np.arange(len(X)), test_size=0.2, random_state=seed, stratify=y)
        X_tr, X_val = X[ti], X[vi]
        y_tr, y_val = y[ti], y[vi]

        clf = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        acc       = accuracy_score(y_val, clf.predict(X_val))
        random_b  = 1.0 / len(train_dsets)

        log.info(f"    Dataset-identity accuracy : {acc:.4f}")
        log.info(f"    Random baseline (1/{len(train_dsets)})    : {random_b:.4f}")
        if acc > 0.70:
            verdict = "LEAKAGE (>70%) — invariance claim overstated"
        elif acc > 0.50:
            verdict = "MIXED (50–70%) — model retains some dataset-specific structure"
        else:
            verdict = "INVARIANCE defensible (<50%)"
        log.info(f"    Decision: {verdict}")

        if not already_done("P3.4a_probe", seed, test_dset):
            log_result("P3.4a_probe", seed, train_dsets, test_dset,
                       "dataset_probe_acc", acc, 0.0)

        # ── P3.4b — UMAP ─────────────────────────────────────────────────────
        log.info(f"\n  P3.4b  UMAP: fold={test_dset}")

        UMAP_PER_DSET = 20000
        umap_embs, umap_ds_labels, umap_atk_labels = [], [], []
        for ds_idx, ds in enumerate(train_dsets):
            g    = load_graph(ds, tier="B", dev=dev)
            embs = _extract_embs(model, g, device, max_feat)
            rng  = np.random.RandomState(seed + ds_idx)
            idx  = rng.choice(len(embs), min(UMAP_PER_DSET, len(embs)), replace=False)
            umap_embs.append(embs[idx])
            umap_ds_labels.extend([ds] * len(idx))
            umap_atk_labels.extend([g.edge_label_type[i] for i in idx])

        X_umap = np.concatenate(umap_embs)
        ds_labels  = np.array(umap_ds_labels)
        atk_labels = np.array(umap_atk_labels)

        log.info(f"    Running UMAP on {len(X_umap)} points (n_neighbors=30, min_dist=0.1)…")
        reducer = umap_mod.UMAP(n_components=2, random_state=seed, n_jobs=1,
                                n_neighbors=30, min_dist=0.1)
        embs_2d = reducer.fit_transform(X_umap)
        log.info(f"    UMAP done: {embs_2d.shape}")

        tab10 = plt.cm.tab10

        # Plot (i): colored by attack type
        unique_types = sorted(set(atk_labels))
        type_to_int  = {t: i for i, t in enumerate(unique_types)}
        fig, ax = plt.subplots(figsize=(8, 6))
        for lbl, idx in type_to_int.items():
            mask = np.array([type_to_int[t] for t in atk_labels]) == idx
            ax.scatter(embs_2d[mask, 0], embs_2d[mask, 1], s=1, alpha=0.3,
                       color=tab10(idx / max(len(type_to_int) - 1, 1)), label=lbl)
        ax.legend(markerscale=6, fontsize=7)
        ax.set_title(f"P3.4b E1.E  fold={test_dset}  colored by attack type")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        fig.tight_layout()
        out = FIGURES_DIR / f"p3_4b_umap_{test_dset}_attack.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        log.info(f"    Saved {out}")

        # Plot (ii): colored by source dataset
        unique_ds = sorted(set(ds_labels))
        ds_to_int = {d: i for i, d in enumerate(unique_ds)}
        ds_colors = ["tab:blue", "tab:orange", "tab:green"]
        fig, ax = plt.subplots(figsize=(8, 6))
        for ds, idx in ds_to_int.items():
            mask = ds_labels == ds
            ax.scatter(embs_2d[mask, 0], embs_2d[mask, 1], s=1, alpha=0.3,
                       color=ds_colors[idx % len(ds_colors)], label=ds)
        ax.legend(markerscale=6, fontsize=7)
        ax.set_title(f"P3.4b E1.E  fold={test_dset}  colored by source dataset")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        fig.tight_layout()
        out = FIGURES_DIR / f"p3_4b_umap_{test_dset}_dataset.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        log.info(f"    Saved {out}")

        log.info("    Attack clusters clean, dataset mixed → invariance real")
        log.info("    Dataset clusters clean, attack scattered → fingerprint leakage")


# ── P3.5 — Scale-invariant feature hybrid ────────────────────────────────────

def run_p3_5_scale_invariant(folds_arg, seed, dev):
    log.info("=== P3.5  Scale-invariant feature hybrid (optional) ===")
    log.info("    Features: bytes_per_packet, packets_per_second, flag_density, 8 tcp_flag_bits")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if folds_arg == "all":
        folds_to_run = ALL_FOLDS
    else:
        target_test = NAME_MAP.get(folds_arg, folds_arg)
        folds_to_run = [f for f in ALL_FOLDS if f["test"] == target_test]

    for fold in folds_to_run:
        test_dset   = fold["test"]
        train_dsets = fold["train"]

        if already_done("P3.5_scale_inv", seed, test_dset):
            log.info(f"  Skipping P3.5 test={test_dset} (already done)")
            continue

        seed_everything(seed)
        t0 = time.time()

        # Load Tier-A, compute scale-invariant features on the fly
        train_graphs_a = [load_graph(ds, tier="A", dev=dev) for ds in train_dsets]
        test_graph_a   = load_graph(test_dset, tier="A", dev=dev)

        train_si = [_build_scale_inv_graph(g) for g in train_graphs_a]
        test_si  = _build_scale_inv_graph(test_graph_a)

        # Combine training graphs (feature dim is uniform = 11)
        combined_si = combine_graphs(train_si)

        # Align test feature dim to training (should both be 11, but be safe)
        max_feat = combined_si.edge_attr.shape[1]
        d = test_si.edge_attr.shape[1]
        if d < max_feat:
            pad = torch.zeros(test_si.edge_attr.shape[0], max_feat - d)
            test_si.edge_attr   = torch.cat([test_si.edge_attr, pad], dim=1)
            test_si.edge_attr_q = torch.cat([test_si.edge_attr_q, pad], dim=1)
        elif d > max_feat:
            test_si.edge_attr   = test_si.edge_attr[:, :max_feat]
            test_si.edge_attr_q = test_si.edge_attr_q[:, :max_feat]

        model, best_state = _train_e1e(combined_si, device, seed)
        save_model("P3.5_scale_inv", seed, test_dset, best_state)

        result  = eval_egraphsage(model, test_si, device=device, use_quantile=True)
        metrics = compute_all_metrics(result["y_true"], result["y_pred"],
                                      y_true_type=test_si.edge_label_type)
        elapsed = time.time() - t0

        log_result("P3.5_scale_inv", seed, train_dsets, test_dset,
                   "mcc", metrics["mcc"], elapsed)
        log_result("P3.5_scale_inv", seed, train_dsets, test_dset,
                   "macro_f1", metrics["macro_f1"], elapsed)

        e1e_refs = {"cic_ids2018": 0.597, "ton_iot": 0.259}
        ref = e1e_refs.get(test_dset)
        ref_str = f"  vs E1.E ref {ref:.3f}: " if ref else ""
        diff    = metrics["mcc"] - ref if ref else 0.0
        verdict = ""
        if ref:
            verdict = ("wins (≥+0.03)" if diff >= 0.03
                       else "comparable (±0.03)" if diff >= -0.03
                       else "worse (<-0.03)")
        log.info(f"  P3.5 fold={test_dset}: MCC={metrics['mcc']:.4f}  "
                 f"macro-F1={metrics['macro_f1']:.4f}  "
                 f"{ref_str}{verdict}")


# ── main ─────────────────────────────────────────────────────────────────────

def _resolve_folds(folds_arg):
    """Accept 'all', a short name like 'cic18', or a full dataset name."""
    if folds_arg == "all":
        return "all"
    return NAME_MAP.get(folds_arg, folds_arg)


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Phase 3 experiments (spex3.md)")
    parser.add_argument("--exp",     required=True,
                        choices=["per_attack", "cic17_single",
                                 "embed_analysis", "scale_invariant", "all"])
    parser.add_argument("--folds",   default="all",
                        help="all | cic18 | ton | lycos | unsw  (used by scale_invariant)")
    parser.add_argument("--models",  nargs="+", default=["cic18", "ton"],
                        help="Fold(s) for embed_analysis: cic18 ton lycos unsw")
    parser.add_argument("--seed",    type=int, default=0)
    parser.add_argument("--dev",     action="store_true", default=True)
    parser.add_argument("--no-dev",  dest="dev", action="store_false")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.exp == "per_attack" or args.exp == "all":
        run_p3_2_per_attack(args.seed, args.dev)

    if args.exp == "cic17_single" or args.exp == "all":
        run_p3_3_cic17_single(args.seed, args.dev)

    if args.exp == "embed_analysis" or args.exp == "all":
        run_p3_4_embed_analysis(args.models, args.seed, args.dev)

    if args.exp == "scale_invariant" or args.exp == "all":
        run_p3_5_scale_invariant(_resolve_folds(args.folds), args.seed, args.dev)


if __name__ == "__main__":
    main()
