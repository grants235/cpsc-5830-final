"""
Results logging to results/results.csv.
Columns: experiment_id, seed, train_datasets, test_dataset, metric, value, wall_clock_sec
"""

import csv
import logging
import os
from pathlib import Path
from typing import Any

import torch

RESULTS_DIR  = Path(__file__).resolve().parents[2] / "results"
RESULTS_FILE = RESULTS_DIR / "results.csv"
MODELS_DIR   = RESULTS_DIR / "models"

COLUMNS = ["experiment_id", "seed", "train_datasets", "test_dataset",
           "metric", "value", "wall_clock_sec"]


def _ensure_csv():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()


def already_done(experiment_id: str, seed: int, test_dataset: str) -> bool:
    """Return True if this (experiment_id, seed, test_dataset) is already in results.csv."""
    if not RESULTS_FILE.exists():
        return False
    with open(RESULTS_FILE, newline="") as f:
        for row in csv.DictReader(f):
            if (row["experiment_id"] == experiment_id
                    and row["seed"] == str(seed)
                    and row["test_dataset"] == test_dataset):
                return True
    return False


def log_result(experiment_id: str, seed: int, train_datasets: list,
               test_dataset: str, metric: str, value: Any,
               wall_clock_sec: float = 0.0):
    _ensure_csv()
    row = {
        "experiment_id":  experiment_id,
        "seed":           seed,
        "train_datasets": "|".join(train_datasets),
        "test_dataset":   test_dataset,
        "metric":         metric,
        "value":          value,
        "wall_clock_sec": wall_clock_sec,
    }
    with open(RESULTS_FILE, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)
    logging.getLogger(__name__).info(
        f"[{experiment_id}] seed={seed} test={test_dataset} {metric}={value:.4f}"
        if isinstance(value, float) else
        f"[{experiment_id}] seed={seed} test={test_dataset} {metric}={value}"
    )


def save_model(experiment_id: str, seed: int, test_dataset: str, obj: Any):
    """Save a trained model (state_dict, sklearn object, or list thereof) to results/models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{experiment_id}_seed{seed}_test{test_dataset}.pt"
    path  = MODELS_DIR / fname
    torch.save(obj, path)
    logging.getLogger(__name__).info(f"Saved model → {path}")


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
