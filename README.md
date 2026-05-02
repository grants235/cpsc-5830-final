# GNN-NIDS: Cross-Dataset Graph Neural Network Intrusion Detection

A PyTorch Geometric implementation of edge-classification GNNs for network intrusion detection, evaluated under **Leave-One-Dataset-Out (LODO)** cross-generalization.  The headline model (**TS-GIB**) combines temporal subgraph context with a variational information bottleneck to reduce dataset-specific bias.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Data Download](#data-download)
4. [Graph Construction](#graph-construction)
5. [Running Models](#running-models)
6. [Model Descriptions](#model-descriptions)
7. [Output Format](#output-format)
8. [Project Structure](#project-structure)

---

## Overview

Network intrusion detection systems (NIDS) trained on one dataset routinely fail to generalize to traffic from different networks or time periods.  This project benchmarks four GNN architectures on four public NetFlow datasets with a strict Leave-One-Dataset-Out protocol: each model is trained on three datasets and evaluated on the held-out fourth.

**Primary metric:** Matthews Correlation Coefficient (MCC) — robust to class imbalance.

**Datasets:**

| Name | Abbrev | Flows | Attack % |
|------|--------|-------|----------|
| LycoS-IDS2017 | `lycos_ids2017` | ~2.8 M | ~20% |
| CSE-CIC-IDS2018 | `cic_ids2018` | ~16 M | ~13% |
| UNSW-NB15 | `unsw_nb15` | ~2.5 M | ~49% |
| ToN-IoT | `ton_iot` | ~37 M | ~79% |

---

## Requirements

Python ≥ 3.10 and the following packages:

```bash
pip install -r requirements.txt
```

PyTorch Geometric requires a separate install step matching your CUDA version.  See [PyG Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) or run:

```bash
pip install torch==2.10.0
pip install torch_geometric==2.7.0
```

---

## Data Download

Download the four datasets and place them in the `data/` directory as shown below.

### LycoS-IDS2017

Source: [LycoS-IDS2017 (Re-labeled CICIDS2017)](https://staff.fnwi.uva.nl/m.m.badinamehrabani/LycoS-IDS2017.csv)

Expected path: `data/lycos-ids2017/LycoS-IDS2017.csv`

### CSE-CIC-IDS2018

Source: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2018.html) — download the NetFlow version `NF-CSE-CIC-IDS2018.csv`.

Expected path: `data/cic-ids2018/NF-CSE-CIC-IDS2018.csv`

### UNSW-NB15

Source: [UNSW Research Data](https://research.unsw.edu.au/projects/unsw-nb15-dataset) — download the four CSV files `UNSW-NB15_1.csv` through `UNSW-NB15_4.csv`.

Expected path: `data/unsw-nb15/UNSW-NB15_1.csv` … `UNSW-NB15_4.csv`

### ToN-IoT

Source: [UNSW ToN-IoT](https://research.unsw.edu.au/projects/toniot-datasets) — download `NF-ToN-IoT-v3.csv`.

Expected path: `data/ton-iot/NF-ToN-IoT-v3.csv`

---

## Graph Construction

Once all raw CSVs are in place, build the PyG graph objects:

```bash
# Build graphs for all datasets (full size — slow, ~1–2 hours)
python scripts/preprocess.py

# Build only dev subsamples (≤2 M rows each — fast, recommended for testing)
python scripts/preprocess.py --dev

# Build a single dataset
python scripts/preprocess.py --datasets lycos_ids2017 unsw_nb15

# Tier-A only (4 shared features) or Tier-B (full per-dataset features)
python scripts/preprocess.py --tier A
python scripts/preprocess.py --tier B   # default
```

Each dataset is converted into a **PyG `Data` object** and saved to `data/processed/`:

| Field | Shape | Description |
|-------|-------|-------------|
| `edge_index` | `[2, E]` | Source/destination IP node indices |
| `edge_attr` | `[E, D]` | Raw Tier-B flow features |
| `edge_attr_q` | `[E, D]` | Quantile-normalized features (per-graph, per-feature) |
| `edge_time` | `[E]` | Flow timestamps in microseconds |
| `edge_label` | `[E]` | Binary labels: 0=benign, 1=attack |
| `x` | `[N, 8]` | Node features (all-ones placeholder) |

**Graph construction details:**
- Nodes are unique IPs (src and dst combined).
- Edges are individual network flows, sorted chronologically.
- Features are quantile-normalized per graph: each feature is ranked and mapped to `[0, 1]`.
- Flows whose attack label has no mapping in the 6-class taxonomy are dropped.

---

## Running Models

First preprocess the data (see above), then:

```bash
python main.py --model <MODEL> [OPTIONS]
```

### Quick Start

```bash
# E-GraphSAGE with raw features, all 4 LODO folds, seed 0 (dev mode)
python main.py --model egraphsage

# E-GraphSAGE structure-only (no features)
python main.py --model egraphsage --no-features

# GIB-EGraphSAGE, sweep β ∈ {0.001, 0.01, 0.1}
python main.py --model gib --beta 0.001 0.01 0.1

# TS-SAGE (temporal, structure-only), full data, single fold
python main.py --model ts-sage --no-dev --folds lycos_ids2017

# TS-GIB, β=0.01, 3 seeds × all 4 folds, full data (headline result)
python main.py --model ts-gib --beta 0.01 --seeds 0 1 2 --no-dev
```

### All Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *(required)* | `egraphsage`, `gib`, `ts-sage`, or `ts-gib` |
| `--beta BETA [BETA ...]` | `0.01` | KL weight β for GIB / TS-GIB. Pass multiple to sweep. |
| `--no-features` | off | Use constant features (structure-only ablation). Not applicable to temporal models. |
| `--seeds SEED [SEED ...]` | `0` | Random seed(s) to run. |
| `--folds DATASET [...]` | all 4 | Restrict evaluation to specific test datasets. |
| `--dev` / `--no-dev` | `--dev` | Use dev subsamples (fast) vs full datasets (accurate). |
| `--delta SECONDS` | `60` | Temporal context window Δ for ts-sage / ts-gib. |
| `--epochs N` | `50` / `20` | Max training epochs (50 for static, 20 for temporal). |
| `--device` | auto | PyTorch device string, e.g. `cuda`, `cpu`, `cuda:1`. |

---

## Model Descriptions

### E-GraphSAGE (`egraphsage`)

Edge-aware variant of GraphSAGE (Lo et al. 2022).  Edge features are first encoded and scatter-aggregated into node states; two SAGE layers propagate messages; the final edge representation `[h_src, h_dst, e_enc]` is classified by an MLP head.

- Default: uses Tier-B quantile-normalized features.
- With `--no-features`: replaces all features with 1.0 (structure-only baseline).
- Primary metric: MCC at threshold 0.5.

### GIB-EGraphSAGE (`gib`)

Same backbone as E-GraphSAGE with a variational information bottleneck (Alemi et al. 2017 / Wu et al. 2020) on the edge embedding.  The 3H embedding is projected to a Gaussian `(μ, log σ)`, then sampled during training and uses `μ` at test time.

- Loss: `CrossEntropy + β × KL`
- β controls the bottleneck strength.  Typical values: `0.001`, `0.01`, `0.1`.
- Primary metric: MCC at threshold 0.5.

### TS-SAGE (`ts-sage`)

Temporal-Subgraph SAGE.  For each query flow `(u, v, t)`, extracts all flows within a Δ-second window before `t` in a 2-hop neighborhood.  Runs SAGE on this local temporal context subgraph, then classifies the query edge via `[h_u, h_v, enc(q_ea)]`.

- Always uses structure-only context (constant edge features).
- `--delta` controls the context window size in seconds.
- Primary metric: calibrated MCC (top-k% threshold at validation attack rate).

### TS-GIB (`ts-gib`)

Extends TS-SAGE with a variational bottleneck on the per-edge embedding.  The bottleneck encourages the model to discard dataset-specific noise and retain transferable attack patterns.

- Loss: `CrossEntropy + β × KL`
- This is the headline model with the best reported cross-dataset generalization.
- Primary metric: calibrated MCC.

---

## Output Format

Results are appended to `results/results.csv`:

```
experiment_id, seed, train_datasets, test_dataset, metric, value, wall_clock_sec
```

Example rows:

```
ts-gib_b0.01_d60, 0, lycos_ids2017|cic_ids2018|unsw_nb15, ton_iot, calibrated_mcc, 0.487, 3421.2
ts-gib_b0.01_d60, 0, lycos_ids2017|cic_ids2018|unsw_nb15, ton_iot, auroc,           0.812, 3421.2
ts-gib_b0.01_d60, 0, lycos_ids2017|cic_ids2018|unsw_nb15, ton_iot, macro_f1,        0.331, 3421.2
egraphsage_raw,   0, cic_ids2018|unsw_nb15|ton_iot,       lycos_ids2017, mcc,        0.421, 512.1
```

Trained model checkpoints (PyTorch state dicts) are saved to `results/models/`.

Runs already present in `results/results.csv` are skipped automatically on re-execution.

---

## Project Structure

```
.
├── main.py                    # Entry point — runs any model via CLI
├── requirements.txt
│
├── data/
│   ├── lycos-ids2017/         # Raw CSVs (not included)
│   ├── cic-ids2018/
│   ├── unsw-nb15/
│   ├── ton-iot/
│   ├── processed/             # Built by preprocess.py (.pt files)
│   └── splits/                # Temporal 80/20 split indices (.json)
│
├── src/
│   ├── data/
│   │   ├── loaders.py         # Per-dataset CSV → canonical DataFrame
│   │   ├── graph_builder.py   # DataFrame → PyG Data, save/load .pt
│   │   ├── feature_aligner.py # Tier-A (4 shared) / Tier-B (full) features
│   │   ├── temporal_subgraph.py  # BFS temporal subgraph extraction
│   │   └── label_map.py       # Native labels → 6-class taxonomy
│   │
│   ├── models/
│   │   ├── egraphsage.py      # EdgeAwareSAGE
│   │   ├── gib_egraphsage.py  # GIB_EGraphSAGE
│   │   ├── temporal_gnn.py    # TemporalEdgeSAGE, TemporalIDGNN, TS_GIB
│   │   ├── baseline_mlp.py    # RandomForest and MLP baselines
│   │   ├── tgn_ids.py         # Temporal Graph Network
│   │   └── moe_ids.py         # Mixture-of-Experts
│   │
│   ├── train/
│   │   ├── train_loops.py     # Training loops with early stopping
│   │   └── eval.py            # Batch inference utilities
│   │
│   └── utils/
│       ├── metrics.py         # MCC, macro-F1, per-class F1
│       ├── logging.py         # results.csv writer, model saver
│       └── seeding.py         # Reproducible seeding
│
├── scripts/
│   ├── preprocess.py          # Build all PyG graphs from raw data
│   ├── run_phase*.py          # Detailed research experiment scripts
│   └── ...
│
└── results/
    ├── results.csv            # All logged metrics
    └── models/                # Saved model checkpoints (.pt)
```

### Label Taxonomy

All datasets are mapped to a shared 6-class taxonomy:

| Class | Description |
|-------|-------------|
| `Benign` | Normal traffic |
| `Reconnaissance` | Scanning, probing |
| `DoS_DDoS` | Denial of service |
| `Injection_Exploit` | SQL injection, exploits |
| `BruteForce` | Password brute-force |
| `Botnet_C2` | Botnet command & control |

For binary classification (the default), all attack classes are merged into label=1.
