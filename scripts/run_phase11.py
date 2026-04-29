#!/usr/bin/env python3
"""
Phase 11: Calibration diagnostic + data analysis.

Reads pre-computed score files from results/inference/*.pt
(produced by extract_scores.py) and generates:

  A.2  results/calibration_table.csv
  A.3  results/figures/oracle_gap_heatmap.png
       results/figures/calibration_scatter_per_fold.png
  A.4  results/calibration_unsupervised.csv  (conditional on A.3)
  A.5  results/figures/per_attack_oracle_f1_heatmap.png
  B.1  results/figures/attack_fingerprint_heatmap.png
  B.2  results/figures/best_method_umap_per_fold.png
  B.3  results/figures/score_histograms.png

Usage:
    python scripts/run_phase11.py [--skip-umap] [--dev]
    python scripts/run_phase11.py --parts A2 A3 A5 B1
"""

import argparse
import copy
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from src.utils.logging import setup_logging, MODELS_DIR

log = logging.getLogger(__name__)

INFERENCE_DIR = Path("results/inference")
FIGURES_DIR   = Path("results/figures")
RESULTS_DIR   = Path("results")

ALL_DATASETS  = ["lycos_ids2017", "cic_ids2018", "unsw_nb15", "ton_iot"]
ATTACK_CLASSES = [
    "Reconnaissance", "DoS_DDoS", "Injection_Exploit",
    "BruteForce", "Botnet_C2",
]

# ── inference file lookup ─────────────────────────────────────────────────────

def build_inference_lookup(inference_dir: Path) -> dict:
    """
    Returns {(method, fold): [file_path, ...]}
    built by reading the metadata stored inside each .pt file.
    """
    lookup: dict = defaultdict(list)
    for f in sorted(inference_dir.glob("*.pt")):
        try:
            data = torch.load(f, weights_only=False)
            key  = (data["method"], data["test_fold"])
            lookup[key].append(f)
        except Exception:
            pass
    return lookup


# ── metric helpers ────────────────────────────────────────────────────────────

def _mcc(y_true, y_pred):
    from sklearn.metrics import matthews_corrcoef
    return float(matthews_corrcoef(y_true, y_pred))


def _auroc(y_true, scores):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


def _auprc(y_true, scores):
    from sklearn.metrics import average_precision_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, scores))
    except Exception:
        return float("nan")


def oracle_mcc(scores: np.ndarray, labels: np.ndarray, n_eval: int = 500):
    """
    O(E log E + n_eval) oracle MCC via cumulative-TP sweep.

    Old approach: 1000 calls to sklearn.matthews_corrcoef, each O(E)
      → O(E × 1000) ≈ 5 billion ops for a 5 M-edge graph.
    New approach: sort once (O(E log E)), then one vectorised numpy pass
      over n_eval candidate cut-points (no Python loop over edges at all).
      → ~50-100× faster for large graphs.
    """
    P  = float(labels.sum())
    N_ = float(len(labels)) - P
    if P == 0 or N_ == 0:
        return 0.0, float(scores.mean())
    if float(scores.min()) == float(scores.max()):
        pred = (scores >= scores[0]).astype(np.int64)
        return _mcc(labels, pred), float(scores[0])

    # Sort descending by score; cum_tp[k] = TP when top-(k+1) predicted positive
    order  = np.argsort(-scores)
    cum_tp = np.cumsum(labels[order].astype(np.float64))

    # Sample n_eval evenly-spaced "number predicted positive" values
    E   = len(scores)
    ks  = np.unique(np.round(np.linspace(0, E, n_eval + 1)).astype(np.int64))
    ks  = ks[(ks >= 0) & (ks <= E)]

    # Vectorised TP / FP / TN / FN for every k
    tp = np.where(ks == 0, 0.0, cum_tp[np.clip(ks - 1, 0, E - 1)])
    tp = np.where(ks == 0, 0.0, tp)
    fp = ks.astype(np.float64) - tp
    tn = N_ - fp
    fn = P  - tp

    d   = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = np.where(d > 0, (tp * tn - fp * fn) / np.sqrt(np.maximum(d, 1e-12)), 0.0)

    best_idx = int(np.argmax(mcc))
    best_k   = int(ks[best_idx])
    best_mcc = float(mcc[best_idx])

    # Threshold = score of the (best_k)-th highest-scored edge
    if best_k == 0:
        best_t = float(scores.max()) + 1e-6
    elif best_k >= E:
        best_t = float(scores.min()) - 1e-6
    else:
        best_t = float(scores[order[best_k]])
    return best_mcc, best_t


def reported_mcc_at(scores, labels, threshold=0.5):
    return _mcc(labels, (scores >= threshold).astype(int))


# ── A.2  Calibration table ────────────────────────────────────────────────────

def _load_inference_file(f: Path) -> dict:
    """Load an inference .pt file, handling both old (string list) and new (int8) formats."""
    data = torch.load(f, weights_only=False)
    # Decode attack_classes: new format stores int8 ids + vocab
    if "attack_class_ids" in data:
        vocab = data.get("attack_class_vocab",
                         ["Benign","Reconnaissance","DoS_DDoS",
                          "Injection_Exploit","BruteForce","Botnet_C2"])
        ids   = np.asarray(data["attack_class_ids"], dtype=np.int8)
        data["attack_classes"] = [vocab[i] if 0 <= i < len(vocab) else "Unknown"
                                  for i in ids]
    # Normalise labels dtype
    data["labels"] = np.asarray(data["labels"], dtype=np.int64)
    return data


def _metrics_for_file(f_str: str):
    """
    Compute calibration metrics for one inference file.
    Top-level function so it can be pickled for ProcessPoolExecutor.
    Returns a row dict, or None on error.
    """
    import numpy as np
    import torch
    from pathlib import Path
    from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

    f = Path(f_str)
    try:
        data = torch.load(f, weights_only=False)
    except Exception as e:
        return {"_error": str(e), "_file": f_str}

    scores = np.asarray(data["scores"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.int64)
    method = data["method"]
    seed   = data["seed"]
    fold   = data["test_fold"]

    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return None

    # reported MCC at 0.5
    preds_05  = (scores >= 0.5).astype(np.int64)
    rep_mcc   = float(matthews_corrcoef(labels, preds_05))

    # oracle MCC — vectorised cumulative-TP sweep
    P  = float(labels.sum())
    N_ = float(len(labels)) - P
    if P > 0 and N_ > 0 and scores.min() != scores.max():
        order  = np.argsort(-scores)
        cum_tp = np.cumsum(labels[order].astype(np.float64))
        E      = len(scores)
        ks     = np.unique(np.round(np.linspace(0, E, 501)).astype(np.int64))
        ks     = ks[(ks >= 0) & (ks <= E)]
        tp = np.where(ks == 0, 0.0, cum_tp[np.clip(ks - 1, 0, E - 1)])
        tp = np.where(ks == 0, 0.0, tp)
        fp = ks.astype(np.float64) - tp
        tn = N_ - fp;  fn = P - tp
        d   = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
        mcc_arr = np.where(d > 0, (tp*tn - fp*fn) / np.sqrt(np.maximum(d, 1e-12)), 0.0)
        best_idx = int(np.argmax(mcc_arr))
        best_k   = int(ks[best_idx])
        orc_mcc  = float(mcc_arr[best_idx])
        best_t   = (float(scores.max())+1e-6 if best_k == 0
                    else float(scores.min())-1e-6 if best_k >= E
                    else float(scores[order[best_k]]))
    else:
        orc_mcc = rep_mcc
        best_t  = 0.5

    # AUROC / AUPRC
    try:
        au_roc = float(roc_auc_score(labels, scores))
    except Exception:
        au_roc = float("nan")
    try:
        au_prc = float(average_precision_score(labels, scores))
    except Exception:
        au_prc = float("nan")

    ppr05 = float(preds_05.mean())
    tpr05 = float(labels[preds_05 == 1].mean()) if preds_05.sum() > 0 else 0.0

    return {
        "method":              method,
        "seed":                seed,
        "test_fold":           fold,
        "n_edges":             len(scores),
        "prevalence":          float(labels.mean()),
        "reported_mcc":        rep_mcc,
        "oracle_mcc":          orc_mcc,
        "gap":                 orc_mcc - rep_mcc,
        "auroc":               au_roc,
        "auprc":               au_prc,
        "pred_pos_rate_at_05": ppr05,
        "true_pos_rate_at_05": tpr05,
        "optimal_threshold":   best_t,
        "reported_threshold":  0.5,
        "_inf_file":           f_str,
    }


def build_calibration_table(inference_dir: Path, workers: int = 4) -> list:
    """
    Build calibration rows in parallel.
    Each file is independent so we fan out to `workers` processes.
    workers=1 falls back to single-process (useful for debugging).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    files = sorted(inference_dir.glob("*.pt"))
    log.info(f"Building calibration table from {len(files)} files "
             f"(workers={workers}) …")

    rows = []
    if workers <= 1:
        for f in files:
            r = _metrics_for_file(str(f))
            if r and "_error" not in r:
                rows.append(r)
                log.info(f"  {r['method']} s={r['seed']} fold={r['test_fold']}: "
                         f"rep={r['reported_mcc']:.3f} oracle={r['oracle_mcc']:.3f} "
                         f"gap={r['gap']:.3f} auroc={r['auroc']:.3f}")
            elif r and "_error" in r:
                log.warning(f"  Error on {Path(r['_file']).name}: {r['_error']}")
        return rows

    futures = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f in files:
            futures[ex.submit(_metrics_for_file, str(f))] = f.name
        for fut in as_completed(futures):
            fname = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                log.warning(f"  Worker error on {fname}: {e}")
                continue
            if r is None:
                continue
            if "_error" in r:
                log.warning(f"  {Path(r['_file']).name}: {r['_error']}")
                continue
            rows.append(r)
            log.info(f"  {r['method']} s={r['seed']} fold={r['test_fold']}: "
                     f"rep={r['reported_mcc']:.3f} oracle={r['oracle_mcc']:.3f} "
                     f"gap={r['gap']:.3f} auroc={r['auroc']:.3f}")

    rows.sort(key=lambda r: (r["method"], r["seed"], r["test_fold"]))
    return rows


def save_calibration_table(rows: list, out_path: Path):
    cols = [
        "method", "seed", "test_fold", "n_edges", "prevalence",
        "reported_mcc", "oracle_mcc", "gap", "auroc", "auprc",
        "pred_pos_rate_at_05", "true_pos_rate_at_05",
        "optimal_threshold", "reported_threshold",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in cols if k in r})
    log.info(f"Saved calibration table → {out_path}  ({len(rows)} rows)")


# ── A.3  Gap analysis figures ─────────────────────────────────────────────────

def _aggregate_rows(rows: list) -> dict:
    """Returns {(method, fold): stats_dict} averaged over seeds."""
    groups: dict = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["test_fold"])].append(r)

    agg = {}
    for key, rs in groups.items():
        agg[key] = {
            "gap_mean":   float(np.mean([r["gap"]          for r in rs])),
            "gap_std":    float(np.std( [r["gap"]          for r in rs])),
            "auroc_mean": float(np.nanmean([r["auroc"]     for r in rs])),
            "mcc_mean":   float(np.mean([r["reported_mcc"] for r in rs])),
            "orc_mean":   float(np.mean([r["oracle_mcc"]   for r in rs])),
            "n":          len(rs),
        }
    return agg


def gap_heatmap(rows: list, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg     = _aggregate_rows(rows)
    methods = sorted(
        {k[0] for k in agg},
        key=lambda m: -float(np.nanmean(
            [agg[(m, f)]["auroc_mean"] for f in ALL_DATASETS if (m, f) in agg] or [0]
        ))
    )
    folds = ALL_DATASETS

    mat = np.full((len(methods), len(folds)), np.nan)
    for i, m in enumerate(methods):
        for j, f in enumerate(folds):
            if (m, f) in agg:
                mat[i, j] = agg[(m, f)]["gap_mean"]

    fig, ax = plt.subplots(figsize=(max(8, len(folds) * 2),
                                     max(6, len(methods) * 0.4)))
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-0.05, vmax=0.30)
    plt.colorbar(im, ax=ax, label="Oracle − Reported MCC (gap)")
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([f.replace("_", "\n") for f in folds], fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=7)
    ax.set_title("Oracle-Reported MCC Gap\n"
                 "(red=threshold-bottlenecked, blue=ranking-bottlenecked)")

    for i in range(len(methods)):
        for j in range(len(folds)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if abs(v) > 0.15 else "black")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out_path}")


def calibration_scatter(rows: list, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    agg     = _aggregate_rows(rows)
    methods = sorted({k[0] for k in agg})

    def _category(m):
        if "dann" in m or "cdan" in m:   return "adversarial"
        if "anomal" in m or "msa" in m:  return "anomaly"
        if "hybrid" in m:                return "hybrid"
        if "lqe" in m or "lze" in m:    return "local_ref"
        if "gib" in m:                   return "gib"
        if m.startswith("B"):            return "baseline"
        return "feature_ablation"

    CAT_COLORS = {
        "adversarial":     "tab:red",
        "anomaly":         "tab:blue",
        "hybrid":          "tab:cyan",
        "local_ref":       "tab:green",
        "gib":             "tab:purple",
        "baseline":        "tab:gray",
        "feature_ablation":"tab:orange",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for ax, fold in zip(axes, ALL_DATASETS):
        fold_data = [(m, agg[(m, fold)]) for m in methods if (m, fold) in agg]
        if not fold_data:
            ax.set_title(fold)
            continue
        for m, v in fold_data:
            cat  = _category(m)
            col  = CAT_COLORS.get(cat, "black")
            size = max(20, v["orc_mean"] * 200)
            ax.scatter(v["auroc_mean"], v["mcc_mean"], c=col, s=size,
                       alpha=0.7, edgecolors="k", linewidths=0.3)
            ax.annotate(m[:14], (v["auroc_mean"], v["mcc_mean"]),
                        textcoords="offset points", xytext=(3, 3), fontsize=4)
        ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("AUROC")
        ax.set_ylabel("Reported MCC")
        ax.set_title(fold)
        ax.grid(True, alpha=0.3)

    legend_els = [Patch(facecolor=c, label=cat) for cat, c in CAT_COLORS.items()]
    fig.legend(handles=legend_els, loc="lower right", ncol=4, fontsize=7)
    fig.suptitle("AUROC vs Reported MCC per Fold\n(point size ∝ oracle MCC)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out_path}")


def gap_summary(rows: list) -> dict:
    threshold_bn = mixed = ranking_bn = 0
    for r in rows:
        g = r["gap"]
        if   g > 0.10:  threshold_bn += 1
        elif g >= 0.05: mixed += 1
        else:           ranking_bn += 1
    total = len(rows)
    log.info(f"\nGap cluster summary (N={total}):")
    log.info(f"  Threshold-bottlenecked (gap>0.10): {threshold_bn}  "
             f"({100*threshold_bn/max(total,1):.1f}%)")
    log.info(f"  Mixed (0.05≤gap≤0.10):             {mixed}  "
             f"({100*mixed/max(total,1):.1f}%)")
    log.info(f"  Ranking-bottlenecked (gap<0.05):   {ranking_bn}  "
             f"({100*ranking_bn/max(total,1):.1f}%)")
    return {"threshold_bottlenecked": threshold_bn, "mixed": mixed,
            "ranking_bottlenecked": ranking_bn, "total": total}


# ── A.4  Unsupervised threshold calibration ───────────────────────────────────

def _mcc_at_thresholds(scores: np.ndarray, labels: np.ndarray,
                        thresholds: list) -> list:
    """
    Compute MCC at every threshold in `thresholds` with ONE sort + ONE
    cumulative-sum pass.  O(E log E + n_thresholds) vs O(E × n_thresholds).
    """
    P  = float(labels.sum())
    N_ = float(len(scores)) - P
    if P == 0 or N_ == 0:
        return [0.0] * len(thresholds)

    order         = np.argsort(-scores)
    sorted_scores = scores[order]                          # descending
    cum_tp        = np.cumsum(labels[order].astype(np.float64))

    results = []
    for t in thresholds:
        # k = number of edges predicted positive (score >= t)
        # In a descending-sorted array, that is the leftmost index where
        # sorted_scores < t, found via searchsorted on the negated array.
        k = int(np.searchsorted(-sorted_scores, -float(t), side="right"))
        if k == 0:
            tp, fp = 0.0, 0.0
        else:
            tp = float(cum_tp[min(k - 1, len(cum_tp) - 1)])
            fp = float(k) - tp
        tn = N_ - fp
        fn = P  - tp
        d  = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        mcc = (tp * tn - fp * fn) / np.sqrt(max(d, 1e-12)) if d > 0 else 0.0
        results.append(float(mcc))
    return results


def _calibrate_file(f_str: str, rep_mcc: float, orc_mcc: float,
                    method: str, seed: int, fold: str) -> list:
    """
    Compute all calibration MCCs for one high-gap inference file.
    Top-level function so ProcessPoolExecutor can pickle it.
    Returns list of row dicts.
    """
    import numpy as np
    import torch
    from pathlib import Path
    from sklearn.mixture import GaussianMixture

    f = Path(f_str)
    if not f.exists():
        return []

    data   = torch.load(f, weights_only=False)
    scores = np.asarray(data["scores"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.int64)
    gap    = orc_mcc - rep_mcc
    prevalence = float(labels.mean())

    # ── compute all calibration thresholds ──────────────────────────────────
    thresholds = {}

    # Otsu: O(256) histogram scan — fast regardless of E
    bins = np.linspace(scores.min(), scores.max(), 256)
    hist, edges = np.histogram(scores, bins=bins)
    total = hist.sum()
    w0 = mu0 = best_between = 0.0
    best_t_otsu = float(edges[0])
    mu_total = float(np.dot(edges[:-1], hist) / max(total, 1))
    for i, h in enumerate(hist):
        p = h / total
        w0 += p
        w1  = 1.0 - w0
        if w0 <= 0 or w1 <= 0:
            continue
        mu0 = (mu0 * (w0 - p) + edges[i] * p) / w0
        mu1 = (mu_total - w0 * mu0) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
        if between > best_between:
            best_between = between
            best_t_otsu  = edges[i]
    thresholds["otsu"] = best_t_otsu

    # GMM: subsample to 50 K for speed (fitting on 5 M samples was the killer)
    try:
        max_fit = 50_000
        fit_sc  = scores if len(scores) <= max_fit else scores[
            np.random.RandomState(0).choice(len(scores), max_fit, replace=False)]
        gm = GaussianMixture(n_components=2, random_state=0, n_init=1)
        gm.fit(fit_sc.reshape(-1, 1))
        thresholds["gmm"] = float(np.mean(gm.means_.flatten()))
    except Exception:
        thresholds["gmm"] = float("nan")

    # Top-k% and prevalence transfer: just percentile lookups
    for k in [10, 20, 30, 40]:
        thresholds[f"topk{k}"] = float(np.percentile(scores, 100 - k))
    thresholds["prevalence_transfer"] = float(
        np.percentile(scores, 100.0 * (1.0 - prevalence)))

    # ── one vectorised MCC pass for all valid thresholds ────────────────────
    names      = list(thresholds.keys())
    tvals      = [thresholds[n] for n in names]
    valid_mask = [not (isinstance(v, float) and np.isnan(v)) for v in tvals]

    # compute only for valid thresholds
    valid_tvals = [v for v, ok in zip(tvals, valid_mask) if ok]

    P  = float(labels.sum())
    N_ = float(len(scores)) - P
    if P > 0 and N_ > 0:
        order         = np.argsort(-scores)
        sorted_scores = scores[order]
        cum_tp        = np.cumsum(labels[order].astype(np.float64))
        mccs_valid = []
        for t in valid_tvals:
            k = int(np.searchsorted(-sorted_scores, -float(t), side="right"))
            tp = float(cum_tp[min(k - 1, len(cum_tp) - 1)]) if k > 0 else 0.0
            fp = float(k) - tp
            tn = N_ - fp;  fn = P - tp
            d  = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
            mccs_valid.append(
                (tp*tn - fp*fn) / np.sqrt(max(d, 1e-12)) if d > 0 else 0.0)
    else:
        mccs_valid = [0.0] * len(valid_tvals)

    # reconstruct full list with nan for invalid
    vi = 0
    mcc_map = {}
    for name, ok in zip(names, valid_mask):
        if ok:
            mcc_map[name] = float(mccs_valid[vi]); vi += 1
        else:
            mcc_map[name] = float("nan")

    # ── build output rows ────────────────────────────────────────────────────
    out = []
    for name, cal_mcc in mcc_map.items():
        rec = (cal_mcc - rep_mcc) / gap if (gap > 0 and not np.isnan(cal_mcc)) \
              else float("nan")
        out.append({
            "method": method, "seed": seed, "test_fold": fold,
            "reported_mcc": rep_mcc, "oracle_mcc": orc_mcc, "gap": gap,
            "cal_method": name, "calibrated_mcc": cal_mcc,
            "oracle_recovery": rec,
        })
    return out


def _resolve_inf_file(row: dict, inf_lookup: dict) -> str:
    """
    Return the inference file path for a calibration row.
    Uses _inf_file if present (set during A.2 in the same run), otherwise
    falls back to inf_lookup keyed by (method, fold).
    """
    cached = row.get("_inf_file", "")
    if cached and Path(cached).is_file():
        return cached
    files = inf_lookup.get((row["method"], row["test_fold"]), [])
    # Pick the file whose seed matches, or the first available
    for f in files:
        data = torch.load(f, weights_only=False)
        if data.get("seed") == row["seed"]:
            return str(f)
    return str(files[0]) if files else ""


def unsupervised_calibration(rows: list, gap_summary_: dict, out_csv: Path,
                              inf_lookup: dict = None, workers: int = 4):
    if gap_summary_["threshold_bottlenecked"] <= 5:
        log.info("A.4: <6 threshold-bottlenecked pairs — skipping")
        return

    from concurrent.futures import ProcessPoolExecutor, as_completed

    inf_lookup = inf_lookup or {}
    high_gap = [r for r in rows if r["gap"] > 0.10]
    log.info(f"A.4: Calibrating {len(high_gap)} high-gap files "
             f"(workers={workers}) …")

    all_rows = []
    if workers <= 1:
        for r in high_gap:
            inf_path = _resolve_inf_file(r, inf_lookup)
            sub = _calibrate_file(inf_path, r["reported_mcc"], r["oracle_mcc"],
                                   r["method"], r["seed"], r["test_fold"])
            for row in sub:
                all_rows.append(row)
                log.info(f"  {row['method']} {row['test_fold']} "
                         f"{row['cal_method']}: mcc={row['calibrated_mcc']:.3f}"
                         f"  rec={row['oracle_recovery']:.3f}")
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in high_gap:
                fut = ex.submit(_calibrate_file,
                                _resolve_inf_file(r, inf_lookup),
                                r["reported_mcc"], r["oracle_mcc"],
                                r["method"], r["seed"], r["test_fold"])
                futures[fut] = r["method"]
            for fut in as_completed(futures):
                try:
                    sub = fut.result()
                except Exception as e:
                    log.warning(f"  Worker error: {e}"); continue
                for row in sub:
                    all_rows.append(row)
                    log.info(f"  {row['method']} {row['test_fold']} "
                             f"{row['cal_method']}: mcc={row['calibrated_mcc']:.3f}"
                             f"  rec={row['oracle_recovery']:.3f}")

    cols = ["method","seed","test_fold","reported_mcc","oracle_mcc","gap",
            "cal_method","calibrated_mcc","oracle_recovery"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(all_rows)
    log.info(f"Saved → {out_csv}  ({len(all_rows)} rows)")

    if all_rows:
        by_cal: dict = defaultdict(list)
        for r in all_rows:
            v = r["oracle_recovery"]
            if not np.isnan(v):
                by_cal[r["cal_method"]].append(v)
        log.info("\nA.4 Recovery by calibration method:")
        for cal, recs in sorted(by_cal.items(), key=lambda x: -np.mean(x[1])):
            m = np.mean(recs)
            log.info(f"  {cal:<22} mean_recovery={m:.3f}  n={len(recs)}")
            if m > 0.5:
                log.info(f"    → >0.5 recovery — publishable calibration result!")


# ── A.5  Per-attack-class oracle F1 ──────────────────────────────────────────

def per_attack_oracle_f1(inference_dir: Path, out_path: Path):
    from sklearn.metrics import f1_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log.info("A.5: Computing per-attack-class oracle F1 …")
    cell_data: dict = defaultdict(list)   # (method, fold, attack_class) → [f1]

    for f in sorted(inference_dir.glob("*.pt")):
        try:
            data = _load_inference_file(f)
        except Exception:
            continue
        scores = np.asarray(data["scores"],  dtype=np.float32)
        labels = data["labels"]
        ac     = np.array(data["attack_classes"])
        method = data["method"]
        fold   = data["test_fold"]

        for cls in ATTACK_CLASSES:
            mask = (ac == cls) | (ac == "Benign")
            if mask.sum() < 10:
                continue
            yt, sc = labels[mask], scores[mask]
            if len(np.unique(yt)) < 2:
                continue
            best_f1 = 0.0
            for t in np.linspace(sc.min(), sc.max(), 200):
                f1v = float(f1_score(yt, (sc >= t).astype(int), zero_division=0))
                best_f1 = max(best_f1, f1v)
            cell_data[(method, fold, cls)].append(best_f1)

    methods  = sorted({k[0] for k in cell_data})
    col_keys = [(fld, cls) for fld in ALL_DATASETS for cls in ATTACK_CLASSES]

    mat = np.full((len(methods), len(col_keys)), np.nan)
    for i, m in enumerate(methods):
        for j, (fld, cls) in enumerate(col_keys):
            vals = cell_data.get((m, fld, cls), [])
            if vals:
                mat[i, j] = float(np.mean(vals))

    if mat.size == 0:
        log.warning("A.5: No data, skipping heatmap")
        return

    fig, ax = plt.subplots(figsize=(max(12, len(col_keys) * 0.6),
                                     max(6, len(methods) * 0.35)))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Oracle F1")
    col_labels = [f"{fld.split('_')[0][:4]}\n{cls[:6]}" for fld, cls in col_keys]
    ax.set_xticks(range(len(col_keys)))
    ax.set_xticklabels(col_labels, fontsize=5, rotation=45, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=6)
    ax.set_title("Per-Attack-Class Oracle F1  (best achievable by threshold sweep)")
    for sep in range(len(ATTACK_CLASSES), len(col_keys), len(ATTACK_CLASSES)):
        ax.axvline(sep - 0.5, color="white", lw=1.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out_path}")


# ── B.1  Attack fingerprint heatmap ──────────────────────────────────────────

def attack_fingerprint_heatmap(out_path: Path, dev: bool = False):
    from src.data.graph_builder import load_graph
    from scipy.spatial.distance import cosine
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log.info("B.1: Computing attack fingerprints …")

    def _fingerprint(g, cls: str):
        ac   = np.array(g.edge_label_type)
        mask = ac == cls
        if mask.sum() < 5:
            return None
        ei  = g.edge_index.numpy()
        et  = g.edge_time.numpy()
        src = ei[0][mask]
        dst = ei[1][mask]
        ts  = et[mask]

        out_deg     = np.bincount(src, minlength=g.num_nodes)
        atk_out_deg = out_deg[np.unique(src)]
        mean_od = float(atk_out_deg.mean())
        std_od  = float(atk_out_deg.std())
        max_od  = float(atk_out_deg.max())

        n_src   = len(np.unique(src))
        n_dst   = len(np.unique(dst))
        n_edges = int(mask.sum())
        density = n_edges / max(n_src * n_dst, 1)

        edge_set = set(zip(src.tolist(), dst.tolist()))
        recip    = sum(1 for u, v in edge_set if (v, u) in edge_set) / max(len(edge_set), 1)

        inter_times = []
        for node in np.unique(src):
            node_ts = np.sort(ts[src == node])
            if len(node_ts) > 1:
                inter_times.extend(np.diff(node_ts).tolist())
        mean_ift = float(np.mean(inter_times)) if inter_times else 0.0
        std_ift  = float(np.std(inter_times))  if inter_times else 0.0

        dst_counts  = np.bincount(dst, minlength=g.num_nodes)
        dst_probs   = dst_counts[dst_counts > 0] / max(dst_counts.sum(), 1)
        dst_div     = float(-np.sum(dst_probs * np.log(dst_probs + 1e-10)))

        return np.array([mean_od, std_od, max_od, density, recip,
                         mean_ift, std_ift, float(n_dst), dst_div],
                        dtype=np.float32)

    fingerprints = {}
    for ds in ALL_DATASETS:
        try:
            g = load_graph(ds, tier="B", dev=dev)
        except FileNotFoundError:
            log.warning(f"  Graph not found: {ds}, skipping")
            continue
        for cls in ATTACK_CLASSES:
            fp = _fingerprint(g, cls)
            if fp is not None:
                fingerprints[(ds, cls)] = fp

    if len(fingerprints) < 2:
        log.warning("B.1: Not enough fingerprints, skipping")
        return

    keys    = sorted(fingerprints.keys())
    n       = len(keys)
    fps     = np.array([fingerprints[k] for k in keys])
    col_max = fps.max(axis=0, keepdims=True)
    col_max[col_max == 0] = 1
    fps_n   = fps / col_max

    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                dist_mat[i, j] = cosine(fps_n[i], fps_n[j])
            except Exception:
                dist_mat[i, j] = 1.0

    labels_ = [f"{ds.split('_')[0][:4]}\n{cls[:8]}" for ds, cls in keys]
    fig, ax = plt.subplots(figsize=(max(10, n * 0.65), max(8, n * 0.65)))
    im = ax.imshow(dist_mat, cmap="coolwarm", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Cosine distance (0=similar, 1=different)")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels_, fontsize=6, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels_, fontsize=6)
    ax.set_title("Attack Subgraph Fingerprint Similarity\n"
                 "(low = structurally similar → transfer expected)")

    # Dataset separators
    ds_cnt = defaultdict(int)
    for ds, _ in keys:
        ds_cnt[ds] += 1
    pos = 0
    for ds in ALL_DATASETS:
        pos += ds_cnt.get(ds, 0)
        if 0 < pos < n:
            ax.axhline(pos - 0.5, color="white", lw=1.5)
            ax.axvline(pos - 0.5, color="white", lw=1.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out_path}")


# ── B.2  Embedding UMAP ───────────────────────────────────────────────────────

def embedding_umap(rows: list, inference_lookup: dict, out_path: Path,
                   dev: bool = False, n_per_ds: int = 5000):
    try:
        import umap as umap_lib
    except ImportError:
        log.warning("B.2: umap-learn not installed, skipping")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from src.models.egraphsage import EdgeAwareSAGE
    from extract_scores import (
        detect_arch, _get_edge_in, _get_hidden, _get_node_in, load_test_graph
    )

    log.info("B.2: Computing UMAP embeddings …")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agg = _aggregate_rows(rows)
    best_per_fold = {}
    for fold in ALL_DATASETS:
        cands = {m: v for (m, f), v in agg.items() if f == fold
                 and not np.isnan(v["auroc_mean"])}
        if cands:
            best_per_fold[fold] = max(cands, key=lambda m: cands[m]["auroc_mean"])

    all_embs, all_ds_labels, all_cls_labels = [], [], []

    for fold, method in best_per_fold.items():
        # Find inference file (for metadata + labels)
        inf_files = inference_lookup.get((method, fold), [])
        if not inf_files:
            log.warning(f"  No inference file for {method}/{fold}, skipping")
            continue

        # Derive model checkpoint path from inference file stem
        inf_file   = inf_files[0]
        ckpt_path  = MODELS_DIR / inf_file.name   # same stem, different dir
        if not ckpt_path.exists():
            log.warning(f"  Model not found: {ckpt_path}, skipping")
            continue

        try:
            sd   = torch.load(ckpt_path, weights_only=True)
        except Exception as e:
            log.warning(f"  Cannot load {ckpt_path}: {e}")
            continue

        arch = detect_arch(sd)
        if arch in ("unknown", "moe"):
            log.info(f"  Skipping UMAP for {method}/{fold} (arch={arch})")
            continue

        edge_in = _get_edge_in(sd, arch)
        hidden  = _get_hidden(sd, arch)
        node_in = _get_node_in(sd, arch, hidden)
        graph   = load_test_graph(fold, edge_in, method, dev)

        if arch == "sage":
            encoder = EdgeAwareSAGE(node_in=node_in, edge_in=edge_in, hidden=hidden)
            encoder.load_state_dict(sd)
        elif arch == "dann":
            encoder = EdgeAwareSAGE(node_in=node_in, edge_in=edge_in, hidden=hidden)
            enc_sd  = {k[len("encoder."):]: v for k, v in sd.items()
                       if k.startswith("encoder.")}
            encoder.load_state_dict(enc_sd)
        elif arch == "gib":
            from src.models.gib_egraphsage import GIB_EGraphSAGE
            bn_dim  = sd["to_dist.weight"].shape[0] // 2
            encoder = GIB_EGraphSAGE(node_in=node_in, edge_in=edge_in,
                                      hidden=hidden, bottleneck_dim=bn_dim)
            encoder.load_state_dict(sd)
        else:
            continue

        # Sample edges
        E   = graph.edge_label.shape[0]
        rng = np.random.RandomState(42)
        idx = np.sort(rng.choice(E, min(n_per_ds, E), replace=False))

        from run_phase6 import _to_local_graph
        encoder.eval().to(device)
        x  = graph.x.to(device)
        ei = graph.edge_index.to(device)
        ea = graph.edge_attr_q.to(device)
        parts = []
        with torch.no_grad():
            for s in range(0, len(idx), 4096):
                ids = idx[s:s + 4096]
                x_b, ei_b, ea_b = _to_local_graph(x, ei, ea, ids, device)
                parts.append(encoder.embed(x_b, ei_b, ea_b).cpu().numpy())
        embs = np.concatenate(parts)

        all_embs.append(embs)
        all_ds_labels.extend([fold] * len(embs))
        all_cls_labels.extend([graph.edge_label_type[i] for i in idx.tolist()])

    if not all_embs:
        log.warning("B.2: No embeddings extracted, skipping UMAP")
        return

    X = np.concatenate(all_embs)
    log.info(f"B.2: Running UMAP on {len(X):,} embeddings …")
    reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    Z       = reducer.fit_transform(X)

    ds_unique  = sorted(set(all_ds_labels))
    cls_unique = sorted(set(all_cls_labels))
    ds_cmap    = plt.cm.get_cmap("tab10", max(len(ds_unique), 1))
    cls_cmap   = plt.cm.get_cmap("Set1",  max(len(cls_unique), 1))
    ds_map     = {d: i for i, d in enumerate(ds_unique)}
    cls_map    = {c: i for i, c in enumerate(cls_unique)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    ax1.scatter(Z[:, 0], Z[:, 1],
                c=[ds_cmap(ds_map[d]) for d in all_ds_labels],
                s=2, alpha=0.4, rasterized=True)
    ax1.legend(handles=[Patch(color=ds_cmap(i), label=d) for d, i in ds_map.items()],
               fontsize=7, markerscale=3)
    ax1.set_title("Coloured by source dataset"); ax1.axis("off")

    ax2.scatter(Z[:, 0], Z[:, 1],
                c=[cls_cmap(cls_map[c]) for c in all_cls_labels],
                s=2, alpha=0.4, rasterized=True)
    ax2.legend(handles=[Patch(color=cls_cmap(i), label=c) for c, i in cls_map.items()],
               fontsize=7, markerscale=3)
    ax2.set_title("Coloured by attack class"); ax2.axis("off")

    fig.suptitle("Pre-classifier Embedding UMAP (best-AUROC method per fold)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out_path}")


# ── B.3  Score histograms ─────────────────────────────────────────────────────

def score_histograms(rows: list, inference_lookup: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log.info("B.3: Building score histograms …")
    agg = _aggregate_rows(rows)
    best_per_fold = {}
    for fold in ALL_DATASETS:
        cands = {m: v for (m, f), v in agg.items() if f == fold
                 and not np.isnan(v["auroc_mean"])}
        if cands:
            best_per_fold[fold] = max(cands, key=lambda m: cands[m]["auroc_mean"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, fold in zip(axes, ALL_DATASETS):
        method = best_per_fold.get(fold)
        if method is None:
            ax.set_title(fold); continue

        inf_files = inference_lookup.get((method, fold), [])
        if not inf_files:
            ax.set_title(f"{fold}\n(no data)"); continue

        data   = _load_inference_file(inf_files[0])
        scores = np.asarray(data["scores"], dtype=np.float32)
        labels = data["labels"]
        benign = scores[labels == 0]
        attack = scores[labels == 1]

        bins = np.linspace(0, 1, 50)
        ax.hist(benign, bins=bins, alpha=0.5, color="tab:blue",
                label=f"Benign (n={len(benign):,})", density=True)
        ax.hist(attack, bins=bins, alpha=0.5, color="tab:red",
                label=f"Attack (n={len(attack):,})", density=True)
        ax.axvline(0.5, color="black", lw=1.5, ls="--", label="threshold=0.5")
        ax.set_xlabel("Score (attack probability)")
        ax.set_ylabel("Density")
        ax.set_title(f"{fold}\n{method}")
        ax.legend(fontsize=7)

    fig.suptitle("Score Distributions: Benign vs Attack on Test Fold", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+",
                        choices=["A2","A3","A4","A5","B1","B2","B3","all"],
                        default=["all"])
    parser.add_argument("--dev",       action="store_true")
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--workers",   type=int, default=4,
                        help="Parallel workers for A.2 calibration table (default 4)")
    args = parser.parse_args()

    setup_logging()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    parts   = set(args.parts)
    run_all = "all" in parts

    # ── A.2: calibration table ────────────────────────────────────────────────
    rows = []
    cal_path = RESULTS_DIR / "calibration_table.csv"
    inf_lookup: dict = {}

    if run_all or "A2" in parts:
        if not INFERENCE_DIR.exists() or not any(INFERENCE_DIR.glob("*.pt")):
            log.error(f"No inference files in {INFERENCE_DIR}. "
                      "Run extract_scores.py first.")
            sys.exit(1)
        rows = build_calibration_table(INFERENCE_DIR, workers=args.workers)
        save_calibration_table(rows, cal_path)
        inf_lookup = build_inference_lookup(INFERENCE_DIR)
    else:
        if cal_path.exists():
            with open(cal_path) as f:
                for r in csv.DictReader(f):
                    rows.append({
                        "method": r["method"], "seed": int(r["seed"]),
                        "test_fold": r["test_fold"],
                        "n_edges": int(r["n_edges"]),
                        "prevalence": float(r["prevalence"]),
                        "reported_mcc": float(r["reported_mcc"]),
                        "oracle_mcc":   float(r["oracle_mcc"]),
                        "gap":          float(r["gap"]),
                        "auroc":        float(r["auroc"]),
                        "auprc":        float(r["auprc"]),
                    })
            log.info(f"Loaded {len(rows)} rows from {cal_path}")
        else:
            log.error("No calibration_table.csv. Run with A2 first.")
            sys.exit(1)
        if INFERENCE_DIR.exists():
            inf_lookup = build_inference_lookup(INFERENCE_DIR)

    # ── A.3 ──────────────────────────────────────────────────────────────────
    summary = {}
    if run_all or "A3" in parts:
        log.info("\n=== A.3 Gap analysis ===")
        gap_heatmap(rows, FIGURES_DIR / "oracle_gap_heatmap.png")
        calibration_scatter(rows, FIGURES_DIR / "calibration_scatter_per_fold.png")
        summary = gap_summary(rows)

    # ── A.4 ──────────────────────────────────────────────────────────────────
    if run_all or "A4" in parts:
        log.info("\n=== A.4 Unsupervised calibration ===")
        if not summary:
            summary = gap_summary(rows)
        unsupervised_calibration(rows, summary,
                                 RESULTS_DIR / "calibration_unsupervised.csv",
                                 inf_lookup=inf_lookup,
                                 workers=args.workers)

    # ── A.5 ──────────────────────────────────────────────────────────────────
    if run_all or "A5" in parts:
        log.info("\n=== A.5 Per-attack-class oracle F1 ===")
        if INFERENCE_DIR.exists():
            per_attack_oracle_f1(INFERENCE_DIR, FIGURES_DIR / "per_attack_oracle_f1_heatmap.png")
        else:
            log.error("Inference dir missing; run extract_scores.py first")

    # ── B.1 ──────────────────────────────────────────────────────────────────
    if run_all or "B1" in parts:
        log.info("\n=== B.1 Attack fingerprint heatmap ===")
        attack_fingerprint_heatmap(FIGURES_DIR / "attack_fingerprint_heatmap.png",
                                    dev=args.dev)

    # ── B.2 ──────────────────────────────────────────────────────────────────
    if (run_all or "B2" in parts) and not args.skip_umap:
        log.info("\n=== B.2 Embedding UMAP ===")
        embedding_umap(rows, inf_lookup, FIGURES_DIR / "best_method_umap_per_fold.png",
                        dev=args.dev)

    # ── B.3 ──────────────────────────────────────────────────────────────────
    if run_all or "B3" in parts:
        log.info("\n=== B.3 Score histograms ===")
        score_histograms(rows, inf_lookup, FIGURES_DIR / "score_histograms.png")

    # ── summary ───────────────────────────────────────────────────────────────
    log.info("\n=== Phase 11 complete ===")
    if rows:
        mccs   = [r["reported_mcc"] for r in rows]
        orcs   = [r["oracle_mcc"]   for r in rows]
        gaps   = [r["gap"]          for r in rows]
        aurocs = [r["auroc"] for r in rows if not np.isnan(r.get("auroc", float("nan")))]
        log.info(f"  Reported MCC: mean={np.mean(mccs):.3f}  std={np.std(mccs):.3f}")
        log.info(f"  Oracle  MCC: mean={np.mean(orcs):.3f}  std={np.std(orcs):.3f}")
        log.info(f"  Mean gap:    {np.mean(gaps):.3f}")
        if aurocs:
            log.info(f"  AUROC:       mean={np.mean(aurocs):.3f}")
        n_tb = sum(1 for g in gaps if g > 0.10)
        n_rb = sum(1 for g in gaps if g < 0.05)
        n_mx = len(gaps) - n_tb - n_rb
        log.info(f"\n  Branch 1 (threshold, gap>0.10): {n_tb}/{len(gaps)}")
        log.info(f"  Branch 3 (mixed, 0.05≤gap≤0.10): {n_mx}/{len(gaps)}")
        log.info(f"  Branch 2 (ranking, gap<0.05):    {n_rb}/{len(gaps)}")


if __name__ == "__main__":
    main()
