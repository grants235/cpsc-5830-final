#!/usr/bin/env python3
"""
Phase 10 (spex10.md): Graph Information Bottleneck for cross-dataset NIDS.

E10.1 — Pairwise CIC18→CIC17, β sweep {0.001, 0.01, 0.1, 1.0}, 1 seed
E10.2 — Pairwise CIC18→CIC17, 3 seeds at best β from E10.1
E10.3 — Per-attack F1 on best E10.2 checkpoint (inference only)
E10.4 — Reverse direction CIC17→CIC18, best β, 1 seed
E10.5 — Full LODO on CIC18 and UNSW folds, best β, 1 seed

Usage:
    python scripts/run_phase10.py --exp e10_1
    python scripts/run_phase10.py --exp e10_2
    python scripts/run_phase10.py --exp e10_3 [--seed 0]
    python scripts/run_phase10.py --exp e10_4
    python scripts/run_phase10.py --exp e10_5 [--seed 0]
    python scripts/run_phase10.py --exp all

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as _tts

from run_phase4 import ALL_FOLDS

from src.utils.logging import setup_logging, log_result, already_done, save_model, MODELS_DIR
from src.utils.seeding import seed_everything
from src.utils.metrics import compute_all_metrics, compute_mcc
from src.data.graph_builder import load_graph, combine_graphs
from src.models.gib_egraphsage import GIB_EGraphSAGE
from src.train.train_loops import _class_weights
from src.train.eval import eval_egraphsage

log = logging.getLogger(__name__)

FIGURES_DIR   = Path("results/figures/phase10")
BETA_SWEEP    = [0.001, 0.01, 0.1, 1.0]
DEFAULT_BETA  = 0.01
BETA_WARMUP   = 5       # epochs to linearly anneal beta from 0 → beta_max
MAX_EPOCHS    = 30
PATIENCE      = 10
MIN_EPOCHS    = 5
BATCH_SIZE    = 2048

# Pairwise CIC17↔CIC18
FOLD_CIC18_TO_CIC17 = {"train": ["cic_ids2018"],   "test": "lycos_ids2017"}
FOLD_CIC17_TO_CIC18 = {"train": ["lycos_ids2017"], "test": "cic_ids2018"}

# LODO folds from ALL_FOLDS for E10.5
LODO_FOLDS_E10 = [
    f for f in ALL_FOLDS if f["test"] in ("cic_ids2018", "unsw_nb15")
]


# ── GIB training loop ─────────────────────────────────────────────────────────

def train_gib(
    model: GIB_EGraphSAGE,
    train_data,
    val_split: dict,
    beta_max: float,
    device: str = "cpu",
    use_quantile: bool = True,
    epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    min_epochs: int = MIN_EPOCHS,
    batch_size: int = BATCH_SIZE,
    warmup_epochs: int = BETA_WARMUP,
) -> dict:
    """
    Train GIB-EGraphSAGE with β-annealing and early stopping on val MCC.

    β is linearly ramped from 0 → beta_max over `warmup_epochs` epochs to
    prevent the KL term from collapsing the encoder before it learns anything.

    Returns best state_dict (by val MCC).
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    cw        = _class_weights(train_data.edge_label, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    edge_attr  = (train_data.edge_attr_q if use_quantile else train_data.edge_attr).to(device)
    x          = train_data.x.to(device)
    edge_index = train_data.edge_index.to(device)
    all_labels = train_data.edge_label

    ti = torch.as_tensor(val_split["train"], dtype=torch.long)
    vi = torch.as_tensor(val_split["val"],   dtype=torch.long)

    best_mcc, best_state, pat_cnt = -2.0, None, 0

    for epoch in range(epochs):
        # Linear β annealing
        beta = beta_max * min(1.0, (epoch + 1) / warmup_epochs)

        model.train()
        perm = ti[torch.randperm(len(ti))]
        ep_loss = 0.0
        n_batches = 0

        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            ea  = edge_attr[idx]
            ei  = edge_index[:, idx]
            yl  = all_labels[idx].to(device)

            logits, kl = model.forward_train(x, ei, ea)
            loss = criterion(logits, yl) + beta * kl
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss  += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        all_preds = []
        with torch.no_grad():
            for start in range(0, len(vi), batch_size):
                idx   = vi[start:start + batch_size]
                ea    = edge_attr[idx]
                ei    = edge_index[:, idx]
                yl    = all_labels[idx].to(device)
                logits = model(x, ei, ea)
                all_preds.append(logits.argmax(1).cpu().numpy())

        val_labels = all_labels[vi].numpy()
        val_preds  = np.concatenate(all_preds)
        val_mcc    = compute_mcc(val_labels, val_preds)

        log.info(f"  epoch {epoch+1:02d}  β={beta:.4f}  "
                 f"loss={ep_loss/max(1, n_batches):.4f}  val_mcc={val_mcc:.4f}")

        if val_mcc > best_mcc:
            best_mcc   = val_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt    = 0
        else:
            if epoch >= min_epochs:
                pat_cnt += 1
                if pat_cnt >= patience:
                    log.info(f"  Early stop at epoch {epoch+1}")
                    break

    log.info(f"  Best val MCC: {best_mcc:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    return best_state


# ── Probe for GIB ─────────────────────────────────────────────────────────────

def _probe_gib(model: GIB_EGraphSAGE, datasets: list, tier: str, dev: bool,
               seed: int, device: str, use_quantile: bool = True,
               max_per_ds: int = 10_000) -> float:
    """
    Linear probe: predict source dataset from GIB bottleneck embeddings (mu).
    datasets: list of dataset name strings, each gets a unique integer label.
    Returns probe accuracy; returns -1.0 if fewer than 2 datasets are provided.
    """
    if len(datasets) < 2:
        log.info("  Probe skipped (fewer than 2 datasets).")
        return -1.0

    all_embs, all_labels = [], []
    model.eval().to(device)

    for ds_idx, ds in enumerate(datasets):
        g    = load_graph(ds, tier=tier, dev=dev)
        ea   = (g.edge_attr_q if use_quantile else g.edge_attr).to(device)
        x_d  = g.x.to(device)
        ei_d = g.edge_index.to(device)
        E    = ei_d.shape[1]

        embs = []
        with torch.no_grad():
            for s in range(0, E, 50_000):
                embs.append(
                    model.embed(x_d, ei_d[:, s:s + 50_000], ea[s:s + 50_000]).cpu().numpy()
                )
        embs_np = np.concatenate(embs)
        rng     = np.random.RandomState(seed)
        idx     = rng.choice(len(embs_np), min(max_per_ds, len(embs_np)), replace=False)
        all_embs.append(embs_np[idx])
        all_labels.extend([ds_idx] * len(idx))

    X = np.concatenate(all_embs)
    y = np.array(all_labels)
    ti, vi = _tts(np.arange(len(X)), test_size=0.2, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
    clf.fit(X[ti], y[ti])
    return accuracy_score(y[vi], clf.predict(X[vi]))


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _load_fold(fold: dict, dev: bool, use_quantile: bool = True):
    """Load and align train+test graphs with real Tier-B features."""
    train_dsets = fold["train"]
    test_dset   = fold["test"]

    train_graphs = [load_graph(ds, tier="B", dev=dev) for ds in train_dsets]
    combined     = combine_graphs(train_graphs)
    test_graph   = load_graph(test_dset, tier="B", dev=dev)

    # Align test feature dim to combined training dim (same as run_exp1.py)
    max_feat = combined.edge_attr.shape[1]
    d = test_graph.edge_attr.shape[1]
    if d < max_feat:
        pad = torch.zeros(test_graph.edge_attr.shape[0], max_feat - d)
        test_graph.edge_attr   = torch.cat([test_graph.edge_attr,   pad], dim=1)
        test_graph.edge_attr_q = torch.cat([test_graph.edge_attr_q, pad], dim=1)
    elif d > max_feat:
        test_graph.edge_attr   = test_graph.edge_attr[:, :max_feat]
        test_graph.edge_attr_q = test_graph.edge_attr_q[:, :max_feat]

    return combined, test_graph


def _make_val_split(combined, seed: int) -> dict:
    n = combined.edge_label.shape[0]
    ti, vi = _tts(np.arange(n), test_size=0.2, random_state=seed,
                  stratify=combined.edge_label.numpy())
    return {"train": ti.tolist(), "val": vi.tolist()}


def _get_best_beta(seeds=(0,)) -> float:
    """Read results.csv and return the β with highest mean MCC across E10.1 seeds."""
    results_path = Path("results/results.csv")
    if not results_path.exists():
        log.warning("  No results.csv; defaulting to β=0.01")
        return DEFAULT_BETA
    beta_vals: dict = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            eid = row["experiment_id"]
            if not eid.startswith("E10.1_gib_b") or row["metric"] != "mcc":
                continue
            if int(row["seed"]) not in seeds:
                continue
            beta_str = eid.split("_b")[-1]
            beta_vals.setdefault(beta_str, []).append(float(row["value"]))
    if not beta_vals:
        log.warning("  No E10.1 results; defaulting to β=0.01")
        return DEFAULT_BETA
    best_key = max(beta_vals, key=lambda k: np.mean(beta_vals[k]))
    best_mean = np.mean(beta_vals[best_key])
    log.info(f"  Best β from E10.1: {best_key}  mean_mcc={best_mean:.4f}")
    return float(best_key)


def _beta_exp_id(beta: float) -> str:
    """Canonical experiment ID string for a given β value."""
    return f"E10.1_gib_b{beta}"


def _print_summary(exp_id: str, seeds, label: str = ""):
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return
    fold_vals: dict = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["experiment_id"] != exp_id or row["metric"] != "mcc":
                continue
            if int(row["seed"]) in seeds:
                fold_vals.setdefault(row["test_dataset"], []).append(float(row["value"]))
    if not fold_vals:
        return
    tag = f"  {exp_id}" + (f" ({label})" if label else "") + " summary:"
    log.info(tag)
    fold_means = []
    for td, vals in sorted(fold_vals.items()):
        m, s = np.mean(vals), np.std(vals)
        fold_means.append(m)
        log.info(f"    {td:<22} mean={m:.4f}  std={s:.4f}  n={len(vals)}")
    log.info(f"  Overall mean MCC: {np.mean(fold_means):.4f}")


# ── Core fold runner ───────────────────────────────────────────────────────────

def _run_gib_fold(exp_id: str, fold: dict, seed: int, beta: float,
                  dev: bool, use_quantile: bool = True,
                  run_probe: bool = True) -> "GIB_EGraphSAGE | None":
    """
    Train and evaluate GIB on one fold at a given β.
    Returns trained model if run (None if skipped because already_done).
    """
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    train_dsets = fold["train"]
    test_dset   = fold["test"]

    if already_done(exp_id, seed, test_dset):
        log.info(f"  Skipping {exp_id} seed={seed} test={test_dset}")
        return None

    seed_everything(seed)
    t0 = time.time()
    log.info(f"\n  [{exp_id}]  β={beta}  train={train_dsets}  test={test_dset}  seed={seed}")

    combined, test_graph = _load_fold(fold, dev, use_quantile)
    val_split = _make_val_split(combined, seed)

    model = GIB_EGraphSAGE(
        node_in=combined.x.shape[1],
        edge_in=combined.edge_attr.shape[1],
    )
    best_state = train_gib(
        model, combined, val_split, beta_max=beta,
        device=device, use_quantile=use_quantile,
    )
    if best_state:
        model.load_state_dict(best_state)
    save_model(exp_id, seed, test_dset, best_state or model.state_dict())

    result  = eval_egraphsage(model, test_graph, device=device, use_quantile=use_quantile)
    metrics = compute_all_metrics(result["y_true"], result["y_pred"],
                                  y_true_type=test_graph.edge_label_type)
    elapsed = time.time() - t0

    log.info(f"  {exp_id} seed={seed} test={test_dset}"
             f"  MCC={metrics['mcc']:.4f}  macro_F1={metrics['macro_f1']:.4f}")
    log_result(exp_id, seed, train_dsets, test_dset, "mcc",      metrics["mcc"],      elapsed)
    log_result(exp_id, seed, train_dsets, test_dset, "macro_f1", metrics["macro_f1"], elapsed)
    log_result(exp_id, seed, train_dsets, test_dset, "beta",     beta,                0.0)
    for cls, f1 in metrics.get("per_class_f1", {}).items():
        log_result(exp_id, seed, train_dsets, test_dset, f"f1_{cls}", f1, elapsed)

    if run_probe and len(train_dsets) >= 2:
        all_probe_dsets = train_dsets
        probe_acc = _probe_gib(model, all_probe_dsets, tier="B", dev=dev,
                               seed=seed, device=device, use_quantile=use_quantile)
        log.info(f"  Probe (train datasets): {probe_acc:.4f}")
        log_result(exp_id, seed, train_dsets, test_dset, "probe_train", probe_acc, 0.0)

    # Always probe train vs test (meaningful even with 1 training source)
    probe_acc_tv = _probe_gib(model, train_dsets + [test_dset], tier="B", dev=dev,
                               seed=seed, device=device, use_quantile=use_quantile)
    log.info(f"  Probe (train+test datasets): {probe_acc_tv:.4f}")
    log_result(exp_id, seed, train_dsets, test_dset, "probe_train_test", probe_acc_tv, 0.0)

    return model


# ── E10.1 — β sweep ───────────────────────────────────────────────────────────

def run_e10_1(seed: int, dev: bool):
    """E10.1: β sweep {0.001, 0.01, 0.1, 1.0} on CIC18→CIC17, single seed."""
    log.info(f"=== E10.1  β sweep  seed={seed}  betas={BETA_SWEEP} ===")

    for beta in BETA_SWEEP:
        exp_id = _beta_exp_id(beta)
        log.info(f"\n--- β = {beta} ---")
        _run_gib_fold(exp_id, FOLD_CIC18_TO_CIC17, seed, beta, dev)

    # Summary
    log.info("\n  E10.1 β sweep summary (CIC18→CIC17):")
    results_path = Path("results/results.csv")
    if results_path.exists():
        beta_rows: dict = {}
        probe_rows: dict = {}
        with open(results_path) as f:
            for row in csv.DictReader(f):
                if not row["experiment_id"].startswith("E10.1_gib_b"):
                    continue
                if int(row["seed"]) != seed or row["test_dataset"] != "lycos_ids2017":
                    continue
                beta_str = row["experiment_id"].split("_b")[-1]
                if row["metric"] == "mcc":
                    beta_rows[beta_str] = float(row["value"])
                elif row["metric"] == "probe_train_test":
                    probe_rows[beta_str] = float(row["value"])
        for b in BETA_SWEEP:
            bs = str(b)
            mcc  = beta_rows.get(bs,   float("nan"))
            prob = probe_rows.get(bs,  float("nan"))
            log.info(f"    β={b:<6}  mcc={mcc:.4f}  probe={prob:.4f}")

    best_beta = _get_best_beta(seeds=(seed,))
    log.info(f"\n  Decision: best β = {best_beta}")
    _apply_decision_rule(best_beta, seed)


def _apply_decision_rule(best_beta: float, seed: int):
    results_path = Path("results/results.csv")
    if not results_path.exists():
        return
    exp_id = _beta_exp_id(best_beta)
    mcc, probe = None, None
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["experiment_id"] != exp_id or int(row["seed"]) != seed:
                continue
            if row["test_dataset"] != "lycos_ids2017":
                continue
            if row["metric"] == "mcc":
                mcc = float(row["value"])
            elif row["metric"] == "probe_train_test":
                probe = float(row["value"])
    if mcc is None:
        return
    if mcc > 0.30 and probe is not None and probe < 0.70:
        log.info("  → Mechanism works: MCC > 0.30 AND probe < 0.70. Proceed to E10.2.")
    elif mcc > 0.30:
        log.info("  → MCC > 0.30 but probe not reduced; method works for wrong reason.")
    else:
        log.info(f"  → Best MCC={mcc:.4f} ≤ 0.30. GIB not helping; consider pivot.")


# ── E10.2 — Multi-seed at best β ──────────────────────────────────────────────

def run_e10_2(seeds, dev: bool):
    """E10.2: 3 seeds at best β from E10.1, CIC18→CIC17."""
    best_beta = _get_best_beta(seeds=(0,))
    exp_id    = "E10.2_gib"
    log.info(f"=== E10.2  multi-seed  β={best_beta}  seeds={seeds} ===")

    for seed in seeds:
        _run_gib_fold(exp_id, FOLD_CIC18_TO_CIC17, seed, best_beta, dev)

    _print_summary(exp_id, seeds, f"GIB β={best_beta} CIC18→CIC17")

    # Multi-seed decision rule
    results_path = Path("results/results.csv")
    mccs = []
    if results_path.exists():
        with open(results_path) as f:
            for row in csv.DictReader(f):
                if (row["experiment_id"] == exp_id and row["metric"] == "mcc"
                        and int(row["seed"]) in seeds):
                    mccs.append(float(row["value"]))
    if mccs:
        m, s = np.mean(mccs), np.std(mccs)
        log.info(f"  Multi-seed: mean={m:.4f}  std={s:.4f}  n={len(mccs)}")
        if m > 0.30 and s < 0.10:
            log.info("  → Robust positive result. Headline.")
        elif m > 0.30:
            log.info(f"  → MCC > 0.30 but std={s:.3f} is high. Report with caveat.")
        else:
            log.info(f"  → Mean MCC {m:.4f} ≤ 0.30.")


# ── E10.3 — Per-attack F1 ─────────────────────────────────────────────────────

def run_e10_3(seed: int, dev: bool):
    """E10.3: Per-attack F1 on best E10.2 GIB checkpoint (inference only)."""
    log.info("=== E10.3  Per-attack F1 on E10.2 GIB ===")
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id    = "E10.2_gib"
    fold      = FOLD_CIC18_TO_CIC17
    test_dset = fold["test"]
    enc_path  = MODELS_DIR / f"{exp_id}_seed{seed}_test{test_dset}.pt"

    if not enc_path.exists():
        log.warning(f"  Missing checkpoint: {enc_path}  (run e10_2 first)")
        return

    combined, test_graph = _load_fold(fold, dev)
    model = GIB_EGraphSAGE(
        node_in=combined.x.shape[1],
        edge_in=combined.edge_attr.shape[1],
    )
    model.load_state_dict(torch.load(enc_path, weights_only=True))

    result  = eval_egraphsage(model, test_graph, device=device, use_quantile=True)
    metrics = compute_all_metrics(result["y_true"], result["y_pred"],
                                  y_true_type=test_graph.edge_label_type)

    log.info(f"\n  fold={test_dset}  MCC={metrics['mcc']:.4f}")
    log.info("  Per-class F1:")
    for cls, f1 in sorted(metrics.get("per_class_f1", {}).items(),
                           key=lambda x: x[1], reverse=True):
        log.info(f"    {cls:<24}  {f1:.4f}")
        log_result("E10.3_per_attack", seed, fold["train"], test_dset, f"f1_{cls}", f1, 0.0)


# ── E10.4 — Reverse direction ─────────────────────────────────────────────────

def run_e10_4(seed: int, dev: bool):
    """E10.4: CIC17→CIC18 at best β from E10.1."""
    best_beta = _get_best_beta(seeds=(0,))
    exp_id    = "E10.4_gib"
    log.info(f"=== E10.4  CIC17→CIC18  β={best_beta}  seed={seed} ===")
    _run_gib_fold(exp_id, FOLD_CIC17_TO_CIC18, seed, best_beta, dev)
    _print_summary(exp_id, [seed], f"GIB β={best_beta} CIC17→CIC18")


# ── E10.5 — Full LODO ─────────────────────────────────────────────────────────

def run_e10_5(seed: int, dev: bool):
    """E10.5: GIB under full LODO on CIC18 and UNSW folds at best β."""
    best_beta = _get_best_beta(seeds=(0,))
    exp_id    = "E10.5_gib_lodo"
    log.info(f"=== E10.5  LODO  β={best_beta}  seed={seed}  folds=CIC18,UNSW ===")

    for fold in LODO_FOLDS_E10:
        _run_gib_fold(exp_id, fold, seed, best_beta, dev, run_probe=True)

    _print_summary(exp_id, [seed], f"GIB β={best_beta} LODO")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Phase 10 (spex10.md)")
    parser.add_argument("--exp", required=True,
                        choices=["e10_1", "e10_2", "e10_3", "e10_4", "e10_5", "all"])
    parser.add_argument("--seeds",  nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seed",   type=int,  default=0,
                        help="Single seed for e10_1, e10_3, e10_4, e10_5")
    parser.add_argument("--dev",    action="store_true", default=True)
    parser.add_argument("--no-dev", dest="dev", action="store_false")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.exp in ("e10_1", "all"):
        run_e10_1(args.seed, args.dev)

    if args.exp in ("e10_2", "all"):
        run_e10_2(args.seeds, args.dev)

    if args.exp in ("e10_3", "all"):
        run_e10_3(args.seed, args.dev)

    if args.exp in ("e10_4", "all"):
        run_e10_4(args.seed, args.dev)

    if args.exp in ("e10_5", "all"):
        run_e10_5(args.seed, args.dev)


if __name__ == "__main__":
    main()
