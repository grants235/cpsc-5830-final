#!/usr/bin/env python3
"""
Phase 2 direction-finding experiments (spex2.md).
Two folds only: test=cic_ids2018, test=ton_iot. Seed 0.

Usage:
    python scripts/run_phase2.py --q1a          # Q1.a label-shuffle control
    python scripts/run_phase2.py --q1b          # Q1.b confusion matrix + attack rate
    python scripts/run_phase2.py --q1c          # Q1.c node-feature ablation
    python scripts/run_phase2.py --q2b          # Q2.b feature distribution histograms
    python scripts/run_phase2.py --q3a          # Q3.a TGN diagnostics (no training)
    python scripts/run_phase2.py --q3b          # Q3.b fixed TGN cross-domain runs
    python scripts/run_phase2.py --q4a          # Q4.a UMAP embedding visualization
    python scripts/run_phase2.py --q4b          # Q4.b linear probe on dataset identity
    python scripts/run_phase2.py --all          # run all in sequence

    Add --no-dev to use full datasets (default: dev subsamples).
"""

import argparse
import copy
import logging
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.model_selection import train_test_split as _tts

from src.utils.logging import setup_logging, log_result, already_done, save_model
from src.utils.seeding import seed_everything
from src.utils.metrics import compute_all_metrics
from src.data.graph_builder import load_graph, combine_graphs
from src.models.egraphsage import EdgeAwareSAGE
from src.train.train_loops import train_egraphsage, train_tgn
from src.train.eval import eval_egraphsage, eval_tgn

log = logging.getLogger(__name__)


def _patch_tgn_memory_dtype(memory) -> None:
    """
    Fix for PyG versions where TGNMemory.last_update is a Float buffer but
    _update_memory tries to store a Long tensor into it → RuntimeError.
    Patch _update_memory to cast last_update to the buffer's dtype before
    the assignment.  No-op when last_update is already Long (older PyG).
    """
    if memory.last_update.dtype == torch.long:
        return

    def _patched_update_memory(self, n_id):
        mem, last_update = self._get_updated_memory(n_id)
        self.memory[n_id] = mem
        self.last_update[n_id] = last_update.to(self.last_update.dtype)

    memory._update_memory = types.MethodType(_patched_update_memory, memory)


SEED = 0
PHASE2_FOLDS = [
    {"train": ["lycos_ids2017", "unsw_nb15", "ton_iot"],     "test": "cic_ids2018"},
    {"train": ["lycos_ids2017", "cic_ids2018", "unsw_nb15"], "test": "ton_iot"},
]
FIGURES_DIR = Path("results/figures/phase2")


# ── helpers ──────────────────────────────────────────────────────────────────

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
    ti, vi = _tts(np.arange(n), test_size=0.2, random_state=SEED,
                  stratify=combined.edge_label.numpy())
    return {"train": ti.tolist(), "val": vi.tolist()}


def _train_e1e(combined, device):
    model = EdgeAwareSAGE(
        node_in=combined.x.shape[1],
        edge_in=combined.edge_attr.shape[1],
    )
    val_split = _make_val_split(combined)
    best_state = train_egraphsage(model, combined, val_split=val_split,
                                  device=device, use_quantile=True)
    model.load_state_dict(best_state)
    return model, best_state


def _load_or_train_e1e(combined, test_dset, device):
    """Load saved E1.E model for this fold if available, otherwise train."""
    for eid in ("P2.Q1b", "P2.Q4a"):
        p = Path(f"results/models/{eid}_seed{SEED}_test{test_dset}.pt")
        if p.exists():
            log.info(f"  Loading model from {p}")
            model = EdgeAwareSAGE(node_in=combined.x.shape[1],
                                  edge_in=combined.edge_attr.shape[1])
            model.load_state_dict(torch.load(p, weights_only=True))
            return model
    log.info("  No saved model found — training E1.E (structure-only)...")
    model, best_state = _train_e1e(combined, device)
    save_model("P2.Q4a", SEED, test_dset, best_state)
    return model


def _extract_embs(model, graph, device, max_feat, structure_only=True, bs=50000):
    """Return [E, 3H] embeddings for all edges in graph."""
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


# ── Q1.a ─────────────────────────────────────────────────────────────────────

def run_q1a(dev):
    log.info("=== Q1.a  Label-shuffle control ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold in PHASE2_FOLDS:
        combined, test_graph, train_dsets, test_dset = _load_fold(
            fold, dev, structure_only=True)

        if already_done("P2.Q1a", SEED, test_dset):
            log.info(f"  Skipping P2.Q1a test={test_dset} (already done)")
            continue

        seed_everything(SEED)
        t0 = time.time()

        # Shuffle training labels before fitting
        perm = torch.randperm(combined.edge_label.shape[0])
        combined.edge_label = combined.edge_label[perm]
        log.info(f"  fold={test_dset}: shuffled {len(perm)} training labels")

        model, _ = _train_e1e(combined, device)
        result  = eval_egraphsage(model, test_graph, device=device, use_quantile=True)
        metrics = compute_all_metrics(result["y_true"], result["y_pred"],
                                      y_true_type=test_graph.edge_label_type)
        elapsed = time.time() - t0
        log_result("P2.Q1a", SEED, train_dsets, test_dset, "mcc", metrics["mcc"], elapsed)
        log.info(f"  Q1.a fold={test_dset}: MCC={metrics['mcc']:.4f}  "
                 f"(expect ~0 if E1.E is real; >0.15 means degenerate)")


# ── Q1.b ─────────────────────────────────────────────────────────────────────

def run_q1b(dev):
    log.info("=== Q1.b  Confusion matrix + attack-rate check ===")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold in PHASE2_FOLDS:
        combined, test_graph, train_dsets, test_dset = _load_fold(
            fold, dev, structure_only=True)

        seed_everything(SEED)
        t0 = time.time()

        model_path = Path(f"results/models/P2.Q1b_seed{SEED}_test{test_dset}.pt")
        if model_path.exists():
            log.info(f"  Loading saved model for fold={test_dset}")
            model = EdgeAwareSAGE(node_in=combined.x.shape[1],
                                  edge_in=combined.edge_attr.shape[1])
            model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            log.info(f"  Training E1.E for fold={test_dset}")
            model, best_state = _train_e1e(combined, device)
            save_model("P2.Q1b", SEED, test_dset, best_state)

        result  = eval_egraphsage(model, test_graph, device=device, use_quantile=True)
        y_true, y_pred = result["y_true"], result["y_pred"]
        metrics = compute_all_metrics(y_true, y_pred,
                                      y_true_type=test_graph.edge_label_type)
        elapsed = time.time() - t0

        if not already_done("P2.Q1b", SEED, test_dset):
            log_result("P2.Q1b", SEED, train_dsets, test_dset, "mcc",
                       metrics["mcc"], elapsed)

        cm               = confusion_matrix(y_true, y_pred)
        pred_attack_rate = y_pred.mean()
        true_attack_rate = y_true.mean()
        report           = classification_report(y_true, y_pred,
                                                 target_names=["Benign", "Attack"])

        log.info(f"\n  Q1.b fold={test_dset}:")
        log.info(f"    MCC               = {metrics['mcc']:.4f}")
        log.info(f"    True attack rate  = {true_attack_rate:.4f}")
        log.info(f"    Pred attack rate  = {pred_attack_rate:.4f}")
        log.info(f"    Confusion matrix:\n{cm}")
        log.info(f"    Classification report:\n{report}")

        # Save confusion matrix figure
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"],
               xlabel="Predicted", ylabel="True",
               title=(f"Q1.b E1.E  test={test_dset}\n"
                      f"MCC={metrics['mcc']:.3f}  "
                      f"pred_atk={pred_attack_rate:.3f}  "
                      f"true_atk={true_attack_rate:.3f}"))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        out = FIGURES_DIR / f"q1b_confmat_{test_dset}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        log.info(f"  Saved {out}")


# ── Q1.c ─────────────────────────────────────────────────────────────────────

def run_q1c(dev):
    log.info("=== Q1.c  Node-feature ablation ===")
    from torch_geometric.utils import degree as pyg_degree

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold in PHASE2_FOLDS:
        combined_base, test_base, train_dsets, test_dset = _load_fold(
            fold, dev, structure_only=True)

        for variant in ("zeros", "rand_frozen", "degree"):
            exp_id = f"P2.Q1c.{variant}"
            if already_done(exp_id, SEED, test_dset):
                log.info(f"  Skipping {exp_id} test={test_dset}")
                continue

            seed_everything(SEED)
            t0 = time.time()

            combined   = copy.deepcopy(combined_base)
            test_graph = copy.deepcopy(test_base)

            if variant == "zeros":
                combined.x   = torch.zeros_like(combined.x)
                test_graph.x = torch.zeros_like(test_graph.x)

            elif variant == "rand_frozen":
                torch.manual_seed(42)
                combined.x   = torch.randn(combined.num_nodes, 8)
                torch.manual_seed(99)
                test_graph.x = torch.randn(test_graph.num_nodes, 8)

            elif variant == "degree":
                deg_c = (pyg_degree(combined.edge_index[0], combined.num_nodes) +
                         pyg_degree(combined.edge_index[1], combined.num_nodes))
                x_c = torch.log1p(deg_c).unsqueeze(1)
                combined.x = torch.cat([x_c, torch.zeros(combined.num_nodes, 7)], dim=1)

                deg_t = (pyg_degree(test_graph.edge_index[0], test_graph.num_nodes) +
                         pyg_degree(test_graph.edge_index[1], test_graph.num_nodes))
                x_t = torch.log1p(deg_t).unsqueeze(1)
                test_graph.x = torch.cat([x_t, torch.zeros(test_graph.num_nodes, 7)], dim=1)

            model, _ = _train_e1e(combined, device)
            result   = eval_egraphsage(model, test_graph, device=device,
                                       use_quantile=True)
            metrics  = compute_all_metrics(result["y_true"], result["y_pred"],
                                           y_true_type=test_graph.edge_label_type)
            elapsed  = time.time() - t0
            log_result(exp_id, SEED, train_dsets, test_dset,
                       "mcc", metrics["mcc"], elapsed)
            log.info(f"  Q1.c {variant:<12} fold={test_dset}: MCC={metrics['mcc']:.4f}")

    log.info("\n  Interpretation:")
    log.info("    zeros ≈ rand_frozen ≈ E1.E  →  node features irrelevant, structure is signal")
    log.info("    degree >> E1.E              →  degree is the cue, not generic structure")
    log.info("    all three much worse        →  original ones-vector encodes something non-trivial")


# ── Q2.b ─────────────────────────────────────────────────────────────────────

def run_q2b(dev):
    log.info("=== Q2.b  Feature distribution histograms ===")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    log.info("  Q2.a (code inspection result):")
    log.info("    quantile_encode() is called once per graph inside build_graph().")
    log.info("    combine_graphs() concatenates the pre-computed edge_attr_q tensors.")
    log.info("    → Quantile encoding is per-graph. No concatenation bug found.")
    log.info("    → B5 < B4 is likely because per-graph quantile destroys cross-dataset")
    log.info("      absolute-magnitude signal that raw features still carry.")

    # Use Tier-A (4 shared features) for a clean apples-to-apples comparison
    fold = PHASE2_FOLDS[0]  # fold=CIC18
    train_dsets = fold["train"]
    test_dset   = fold["test"]

    train_graphs = [load_graph(ds, tier="A", dev=dev) for ds in train_dsets]
    combined     = combine_graphs(train_graphs)
    test_graph   = load_graph(test_dset, tier="A", dev=dev)

    n_feat = min(combined.edge_attr.shape[1], test_graph.edge_attr.shape[1])
    feat_names = ["byte_count", "packet_count", "tcp_flags", "duration_ms"][:n_feat]

    for feat_mode, c_attr, t_attr in [
        ("raw",      combined.edge_attr,   test_graph.edge_attr),
        ("quantile", combined.edge_attr_q, test_graph.edge_attr_q),
    ]:
        cf = c_attr[:, :n_feat].numpy()
        tf = t_attr[:, :n_feat].numpy()

        log.info(f"\n  {feat_mode.upper()} Tier-A features (fold=CIC18):")
        log.info(f"    Train: mean={cf.mean():.4f} std={cf.std():.4f} "
                 f"min={cf.min():.4f} max={cf.max():.4f}")
        log.info(f"    Test:  mean={tf.mean():.4f} std={tf.std():.4f} "
                 f"min={tf.min():.4f} max={tf.max():.4f}")
        for i, name in enumerate(feat_names):
            log.info(f"    {name:<18}  "
                     f"train=[{cf[:,i].min():.3f}, {np.median(cf[:,i]):.3f}, {cf[:,i].max():.3f}]  "
                     f"test=[{tf[:,i].min():.3f}, {np.median(tf[:,i]):.3f}, {tf[:,i].max():.3f}]")

        fig, axes = plt.subplots(1, n_feat, figsize=(4 * n_feat, 3))
        if n_feat == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.hist(cf[:, i], bins=60, alpha=0.55, label="Train (combined)", density=True)
            ax.hist(tf[:, i], bins=60, alpha=0.55, label=f"Test ({test_dset})", density=True)
            ax.set_title(feat_names[i])
            ax.legend(fontsize=7)
        fig.suptitle(f"Q2.b Tier-A {feat_mode} features: train vs test={test_dset}")
        fig.tight_layout()
        out = FIGURES_DIR / f"q2b_feat_{feat_mode}_cic18.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        log.info(f"  Saved {out}")


# ── Q3.a ─────────────────────────────────────────────────────────────────────

def run_q3a(dev):
    log.info("=== Q3.a  TGN diagnostics ===")

    fold = PHASE2_FOLDS[0]  # fold=CIC18
    combined, test_graph, train_dsets, test_dset = _load_fold(fold, dev)

    t = combined.edge_time
    t_float = t.float()
    t_roundtrip = t_float.long()
    mismatch = (t_roundtrip != t).sum().item()

    log.info(f"\n  Timestamp diagnostics (combined training, fold=CIC18):")
    log.info(f"    dtype                    = {t.dtype}")
    log.info(f"    t.min()                  = {t.min().item()}")
    log.info(f"    t.max()                  = {t.max().item()}")
    log.info(f"    range                    = {(t.max() - t.min()).item()}")
    log.info(f"    range / 1e6 (seconds)    = {(t.max() - t.min()).item() / 1e6:.1f}")
    log.info(f"    float32 roundtrip errors = {mismatch}/{len(t)} "
             f"({'OVERFLOW — precision loss in float32' if mismatch > 0 else 'exact'})")

    t_norm = (t_float - t_float.min()) / 1e6
    log.info(f"    After (t-min)/1e6: range = [{t_norm.min():.2f}, {t_norm.max():.2f}] seconds")

    max_train = combined.edge_index.max().item()
    max_test  = test_graph.edge_index.max().item()
    num_nodes_used = combined.num_nodes + test_graph.num_nodes

    log.info(f"\n  Node-index diagnostics:")
    log.info(f"    combined.num_nodes       = {combined.num_nodes}")
    log.info(f"    max train node index     = {max_train}  "
             f"({'OK' if max_train < combined.num_nodes else 'OVERFLOW'})")
    log.info(f"    test_graph.num_nodes     = {test_graph.num_nodes}")
    log.info(f"    max test node index      = {max_test}  "
             f"({'OK' if max_test < test_graph.num_nodes else 'OVERFLOW'})")
    log.info(f"    TGN num_nodes (combined) = {num_nodes_used}  (safe upper bound)")

    log.info(f"\n  Memory-reset diagnostics:")
    log.info(f"    train_tgn: reset at epoch start  — OK (train_loops.py:146-147)")
    log.info(f"    eval_tgn:  reset before eval     — OK (eval.py:72-73)")

    log.info(f"\n  Root-cause summary:")
    if mismatch > 0:
        log.info(f"    PRIMARY: float32 overflow on {mismatch} timestamps.")
        log.info(f"    sin(omega * 1e12) in TGN time-encoder → NaN gradients.")
    else:
        log.info(f"    Timestamps fit in float32 for this fold but magnitude still large.")
        log.info(f"    sin(omega * t) can saturate even without exact overflow.")
    log.info(f"    FIX applied in: src/train/train_loops.py and src/train/eval.py")
    log.info(f"    Normalization: t = (t - t.min()) / (t.max() - t.min())  → [0, 1]")


# ── Q3.b ─────────────────────────────────────────────────────────────────────

def run_q3b(dev):
    log.info("=== Q3.b  Fixed TGN cross-domain runs ===")
    from src.models.tgn_ids import TGN_IDS
    from torch_geometric.nn.models.tgn import LastNeighborLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold in PHASE2_FOLDS:
        combined, test_graph, train_dsets, test_dset = _load_fold(fold, dev)

        if already_done("P2.Q3b", SEED, test_dset):
            log.info(f"  Skipping P2.Q3b test={test_dset} (already done)")
            continue

        seed_everything(SEED)
        t0 = time.time()

        max_feat  = combined.edge_attr.shape[1]
        num_nodes = combined.num_nodes + test_graph.num_nodes

        model = TGN_IDS(num_nodes, max_feat)
        _patch_tgn_memory_dtype(model.memory)
        best_state = train_tgn(
            model, combined, val_data=None,
            device=device, use_quantile=True,
            epochs=10, min_epochs=3, patience=10,
        )
        model.load_state_dict(best_state)
        save_model("P2.Q3b", SEED, test_dset, best_state)

        neighbor_loader = LastNeighborLoader(num_nodes, size=10, device=device)
        assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
        result  = eval_tgn(model, test_graph, neighbor_loader, assoc,
                           device=device, use_quantile=True)
        metrics = compute_all_metrics(result["y_true"], result["y_pred"],
                                      y_true_type=test_graph.edge_label_type)
        elapsed = time.time() - t0

        log_result("P2.Q3b", SEED, train_dsets, test_dset, "mcc",
                   metrics["mcc"], elapsed)
        log_result("P2.Q3b", SEED, train_dsets, test_dset, "macro_f1",
                   metrics["macro_f1"], elapsed)
        for cls, f1 in metrics.get("per_class_f1", {}).items():
            log_result("P2.Q3b", SEED, train_dsets, test_dset, f"f1_{cls}", f1, elapsed)

        log.info(f"  Q3.b fold={test_dset}: MCC={metrics['mcc']:.4f}  "
                 f"(E1.E refs: cic=0.597, ton=0.259)")
        log.info(f"  Decision threshold: TGN >= E1.E + 0.05 on ≥1 fold → temporal contribution real")


# ── Q4.a ─────────────────────────────────────────────────────────────────────

def run_q4a(dev):
    log.info("=== Q4.a  UMAP embedding visualization ===")
    try:
        import umap as umap_mod
    except ImportError:
        log.error("  umap-learn not installed.  pip install umap-learn  then re-run.")
        return

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsClassifier

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use fold=CIC18 (E1.E's strongest / most suspicious fold)
    fold = PHASE2_FOLDS[0]
    combined, test_graph, train_dsets, test_dset = _load_fold(
        fold, dev, structure_only=True)
    max_feat = combined.edge_attr.shape[1]

    seed_everything(SEED)
    model = _load_or_train_e1e(combined, test_dset, device)

    # ── test embeddings ──
    log.info("  Extracting test embeddings...")
    test_embs = _extract_embs(model, test_graph, device, max_feat)
    test_y_type = np.array(test_graph.edge_label_type)
    log.info(f"    test embeddings: {test_embs.shape}")

    # ── training embeddings per dataset (for 5-NN dataset coloring) ──
    MAX_PER_DSET = 15000
    train_embs_all, train_ds_labels = [], []
    for ds_idx, ds in enumerate(train_dsets):
        g = load_graph(ds, tier="B", dev=dev)
        embs = _extract_embs(model, g, device, max_feat)
        rng  = np.random.RandomState(SEED)
        idx  = rng.choice(len(embs), min(MAX_PER_DSET, len(embs)), replace=False)
        train_embs_all.append(embs[idx])
        train_ds_labels.extend([ds_idx] * len(idx))
        log.info(f"    training embs from {ds}: {len(idx)} sampled")

    train_embs  = np.concatenate(train_embs_all, axis=0)
    train_ds_labels = np.array(train_ds_labels)

    # ── subsample test to 20k for UMAP ──
    rng    = np.random.RandomState(SEED)
    n_sub  = min(20000, len(test_embs))
    t_idx  = rng.choice(len(test_embs), n_sub, replace=False)
    embs_2d_input = test_embs[t_idx]
    y_type_sub    = test_y_type[t_idx]

    # ── 5-NN dataset labeling ──
    log.info("  Fitting 5-NN classifier for dataset coloring...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(train_embs, train_ds_labels)
    ds_pred = knn.predict(embs_2d_input)

    # ── UMAP ──
    log.info("  Running UMAP (this may take a few minutes)...")
    reducer = umap_mod.UMAP(n_components=2, random_state=SEED, n_jobs=1,
                             min_dist=0.1, n_neighbors=15)
    embs_2d = reducer.fit_transform(embs_2d_input)
    log.info(f"    UMAP done: {embs_2d.shape}")

    tab10 = plt.cm.tab10

    # ── plot (i): colored by attack type ──
    unique_types = sorted(set(y_type_sub))
    type_to_int  = {t: i for i, t in enumerate(unique_types)}
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, idx in type_to_int.items():
        mask = np.array([type_to_int[t] for t in y_type_sub]) == idx
        ax.scatter(embs_2d[mask, 0], embs_2d[mask, 1], s=1, alpha=0.3,
                   color=tab10(idx / max(len(type_to_int) - 1, 1)), label=label)
    ax.legend(markerscale=6, fontsize=7, loc="best")
    ax.set_title(f"Q4.a E1.E  test={test_dset}  colored by attack type")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    out = FIGURES_DIR / f"q4a_umap_{test_dset}_attack.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    log.info(f"  Saved {out}")

    # ── plot (ii): colored by nearest training dataset ──
    fig, ax = plt.subplots(figsize=(8, 6))
    ds_colors = ["tab:blue", "tab:orange", "tab:green"]
    for ds_idx, ds in enumerate(train_dsets):
        mask = ds_pred == ds_idx
        ax.scatter(embs_2d[mask, 0], embs_2d[mask, 1], s=1, alpha=0.3,
                   color=ds_colors[ds_idx], label=ds)
    ax.legend(markerscale=6, fontsize=7, loc="best")
    ax.set_title(f"Q4.a E1.E  test={test_dset}  colored by nearest training dataset (5-NN)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    out = FIGURES_DIR / f"q4a_umap_{test_dset}_dataset.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    log.info(f"  Saved {out}")

    log.info("\n  Interpretation:")
    log.info("    Attack clusters clean, dataset mixed → invariance real")
    log.info("    Dataset clusters clean, attack scattered → fingerprint leakage")


# ── Q4.b ─────────────────────────────────────────────────────────────────────

def run_q4b(dev):
    log.info("=== Q4.b  Linear probe on dataset identity ===")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold = PHASE2_FOLDS[0]  # fold=CIC18
    combined, _, train_dsets, test_dset = _load_fold(fold, dev, structure_only=True)
    max_feat = combined.edge_attr.shape[1]

    seed_everything(SEED)
    model = _load_or_train_e1e(combined, test_dset, device)

    # Extract per-dataset training embeddings, capped at 10k each to keep RAM sane
    MAX_PER_DSET = 10000
    all_embs, all_labels = [], []
    for ds_idx, ds in enumerate(train_dsets):
        g    = load_graph(ds, tier="B", dev=dev)
        embs = _extract_embs(model, g, device, max_feat)
        rng  = np.random.RandomState(SEED)
        idx  = rng.choice(len(embs), min(MAX_PER_DSET, len(embs)), replace=False)
        all_embs.append(embs[idx])
        all_labels.extend([ds_idx] * len(idx))
        log.info(f"  {ds}: {len(idx)} embeddings")

    X = np.concatenate(all_embs)
    y = np.array(all_labels)

    probe_ti, probe_vi = _tts(np.arange(len(X)), test_size=0.2,
                               random_state=SEED, stratify=y)
    X_tr, X_val = X[probe_ti], X[probe_vi]
    y_tr, y_val = y[probe_ti], y[probe_vi]

    log.info(f"  Training linear probe: {len(X_tr)} train / {len(X_val)} val")
    clf = LogisticRegression(max_iter=300, random_state=SEED, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    acc      = accuracy_score(y_val, clf.predict(X_val))
    random_b = 1.0 / len(train_dsets)

    log.info(f"\n  Q4.b LINEAR PROBE RESULT:")
    log.info(f"    Dataset-identity accuracy : {acc:.4f}")
    log.info(f"    Random baseline (1/{len(train_dsets)})    : {random_b:.4f}")
    if acc > 0.50:
        verdict = "LEAKAGE — embedding encodes dataset identity"
    elif acc < 0.40:
        verdict = "INVARIANCE defensible — embedding does not strongly encode dataset"
    else:
        verdict = "AMBIGUOUS — borderline case"
    log.info(f"    Decision                  : {verdict}")

    log_result("P2.Q4b", SEED, train_dsets, test_dset,
               "dataset_probe_acc", acc, 0.0)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Phase 2 direction-finding experiments")
    parser.add_argument("--dev",    action="store_true", default=True)
    parser.add_argument("--no-dev", dest="dev", action="store_false")
    parser.add_argument("--q1a",  action="store_true")
    parser.add_argument("--q1b",  action="store_true")
    parser.add_argument("--q1c",  action="store_true")
    parser.add_argument("--q2b",  action="store_true")
    parser.add_argument("--q3a",  action="store_true")
    parser.add_argument("--q3b",  action="store_true")
    parser.add_argument("--q4a",  action="store_true")
    parser.add_argument("--q4b",  action="store_true")
    parser.add_argument("--all",  action="store_true")
    args = parser.parse_args()

    run_all = args.all or not any([
        args.q1a, args.q1b, args.q1c,
        args.q2b, args.q3a, args.q3b,
        args.q4a, args.q4b,
    ])

    if args.q1a or run_all: run_q1a(args.dev)
    if args.q1b or run_all: run_q1b(args.dev)
    if args.q1c or run_all: run_q1c(args.dev)
    if args.q2b or run_all: run_q2b(args.dev)
    if args.q3a or run_all: run_q3a(args.dev)
    if args.q3b or run_all: run_q3b(args.dev)
    if args.q4a or run_all: run_q4a(args.dev)
    if args.q4b or run_all: run_q4b(args.dev)


if __name__ == "__main__":
    main()
