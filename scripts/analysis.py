#!/usr/bin/env python3
"""
Generate paper figures from results/results.csv.
Produces (in order):
  8.1 Main results table (MCC)
  8.2 Per-attack transferability heatmap
  8.3 Ablation ladder
  8.4 Transfer difficulty hierarchy
  8.5 Embedding t-SNE (if --embeddings flag)

Usage:
    python scripts/analysis.py [--out results/figures] [--embeddings]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logging import setup_logging

log = logging.getLogger(__name__)

RESULTS_FILE = Path(__file__).resolve().parents[1] / "results" / "results.csv"

METHOD_ORDER = ["B1_RF", "B4_raw", "B5_quant", "E1.C", "E1.E",
                "E2.B", "E3.B", "B3_within"]
METHOD_LABELS = {
    "B1_RF":    "B1 RF (Tier-A)",
    "B4_raw":   "B4 SAGE raw",
    "B5_quant": "B5 SAGE quant",
    "E1.C":     "E1.C RFE Tier-B",
    "E1.E":     "E1.E struct-only",
    "E2.B":     "E2.B TGN",
    "E3.B":     "E3.B MoE",
    "B3_within":"B3 within (UB)",
}
DATASET_ORDER = ["lycos_ids2017", "cic_ids2018", "unsw_nb15", "ton_iot"]
DATASET_LABELS = {
    "lycos_ids2017": "CIC17",
    "cic_ids2018":   "CIC18",
    "unsw_nb15":     "UNSW",
    "ton_iot":       "ToN",
}
ATTACK_CLASSES = ["Reconnaissance", "DoS_DDoS", "Injection_Exploit", "BruteForce", "Botnet_C2"]

DIFFICULTY = {
    "easy":   [("lycos_ids2017", "cic_ids2018"), ("cic_ids2018", "lycos_ids2017")],
    "medium": [("unsw_nb15", "lycos_ids2017"), ("unsw_nb15", "cic_ids2018")],
    "hard":   [("ton_iot", "lycos_ids2017"), ("ton_iot", "cic_ids2018"),
               ("ton_iot", "unsw_nb15")],
}


def load_results() -> pd.DataFrame:
    if not RESULTS_FILE.exists():
        log.error(f"No results file at {RESULTS_FILE}")
        return pd.DataFrame()
    df = pd.read_csv(RESULTS_FILE)
    return df


def make_main_table(df: pd.DataFrame, out_dir: Path):
    mcc_df = df[df["metric"] == "mcc"].copy()
    pivot = (
        mcc_df.groupby(["experiment_id", "test_dataset"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    pivot["mean_std"] = pivot.apply(
        lambda r: f"{r['mean']:.3f}±{r['std']:.3f}", axis=1
    )
    table = pivot.pivot(index="experiment_id", columns="test_dataset",
                        values="mean_std")
    table = table.reindex(index=METHOD_ORDER, columns=DATASET_ORDER, fill_value="—")
    table.index   = [METHOD_LABELS.get(m, m) for m in table.index]
    table.columns = [DATASET_LABELS.get(d, d) for d in table.columns]

    print("\n=== Main Results Table (MCC mean±std) ===")
    print(table.to_string())

    # Also save as CSV
    table.to_csv(out_dir / "table_main.csv")
    log.info(f"Saved table_main.csv")


def make_heatmap(df: pd.DataFrame, out_dir: Path, method: str = "E2.B"):
    f1_df = df[(df["experiment_id"] == method) &
               df["metric"].str.startswith("f1_")].copy()
    f1_df["attack"] = f1_df["metric"].str.replace("f1_", "")
    pivot = f1_df.groupby(["attack", "test_dataset"])["value"].mean().unstack()
    pivot = pivot.reindex(index=ATTACK_CLASSES,
                          columns=DATASET_ORDER).rename(columns=DATASET_LABELS)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot.astype(float), annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title(f"Per-Attack F1 ({METHOD_LABELS.get(method, method)})")
    ax.set_xlabel("Held-out Dataset")
    ax.set_ylabel("Attack Type")
    plt.tight_layout()
    out = out_dir / "heatmap_per_attack.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info(f"Saved {out}")


def make_ablation_ladder(df: pd.DataFrame, out_dir: Path):
    ladder_methods = ["B4_raw", "B5_quant", "E1.C", "E2.B", "E3.B"]
    mcc_df = df[(df["metric"] == "mcc") & df["experiment_id"].isin(ladder_methods)]
    avg = mcc_df.groupby(["experiment_id", "test_dataset"])["value"].mean().reset_index()

    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(12, 4), sharey=True)
    for ax, ds in zip(axes, DATASET_ORDER):
        sub = avg[avg["test_dataset"] == ds].set_index("experiment_id")
        vals = [sub.loc[m, "value"] if m in sub.index else float("nan")
                for m in ladder_methods]
        labels = [METHOD_LABELS.get(m, m) for m in ladder_methods]
        ax.bar(range(len(ladder_methods)), vals, color="steelblue")
        ax.set_xticks(range(len(ladder_methods)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_ylim(0, 1)
        if ax == axes[0]:
            ax.set_ylabel("MCC")

    fig.suptitle("Ablation Ladder (MCC, mean over 3 seeds)")
    plt.tight_layout()
    out = out_dir / "ablation_ladder.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info(f"Saved {out}")


def make_difficulty_hierarchy(df: pd.DataFrame, out_dir: Path):
    mcc_df = df[df["metric"] == "mcc"].copy()
    mcc_df["train_test"] = list(zip(
        mcc_df["test_dataset"],
        mcc_df["train_datasets"].str.split("|").apply(lambda x: x[0]),
    ))
    records = []
    for method in METHOD_ORDER[:-1]:  # exclude upper bound
        for diff, pairs in DIFFICULTY.items():
            sub = mcc_df[(mcc_df["experiment_id"] == method)]
            vals = sub[sub["test_dataset"].isin([p[0] for p in pairs])]["value"].values
            if len(vals) == 0:
                continue
            records.append({
                "method": METHOD_LABELS.get(method, method),
                "difficulty": diff,
                "mcc": vals.mean(),
            })
    rec_df = pd.DataFrame(records)
    if rec_df.empty:
        log.warning("Not enough data for difficulty hierarchy plot")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = rec_df.pivot(index="method", columns="difficulty", values="mcc")
    pivot.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_ylabel("Avg MCC")
    ax.set_title("Transfer Difficulty Hierarchy")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    out = out_dir / "difficulty_hierarchy.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info(f"Saved {out}")


def make_gating_plot(df: pd.DataFrame, out_dir: Path):
    """Gate weight stacked bar if E3.B produced gating data."""
    # Placeholder: gate weights are logged separately; if gate_w_* metrics exist, plot
    gate_df = df[df["metric"].str.startswith("gate_w_")]
    if gate_df.empty:
        log.info("No gating weight data found; skipping gating plot")
        return
    # (filled in if gating weights are added to logging later)


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "figures"))
    parser.add_argument("--embeddings", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results()
    if df.empty:
        log.warning("No results to analyze. Run experiments first.")
        return

    make_main_table(df, out_dir)
    make_heatmap(df, out_dir)
    make_ablation_ladder(df, out_dir)
    make_difficulty_hierarchy(df, out_dir)
    make_gating_plot(df, out_dir)

    if args.embeddings:
        log.info("Embedding analysis not yet automated — run interactively from a notebook.")


if __name__ == "__main__":
    main()
