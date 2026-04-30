#!/usr/bin/env python3
"""
Phase 11 A.1: Score extraction across all checkpoints.

Memory design (26 GB budget):
  • Checkpoints are sorted by test_fold so each test graph is loaded ONCE
    and reused for all models that test on that fold, then freed completely.
  • ip_to_idx (can be a huge Python dict of IP strings) is deleted immediately
    after graph load – it is never needed for inference.
  • attack_classes stored as int8 indices (+ tiny vocab), not a Python list of
    millions of strings.
  • _infer_sage/_infer_dann use local-subgraph batching with a PRE-ALLOCATED
    assoc buffer, so no N_total_nodes tensor is re-created every batch.
  • scores/labels/classes are explicitly deleted right after saving.

Usage (on remote server):
    python scripts/extract_scores.py [--dev] [--overwrite]
    python scripts/extract_scores.py --method E1.A   # single method
"""

import argparse
import copy
import gc
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from src.data.graph_builder import (
    load_graph, load_split, quantile_encode, PROCESSED_DIR
)
from src.models.egraphsage import EdgeAwareSAGE
from src.models.gib_egraphsage import GIB_EGraphSAGE
from src.models.moe_ids import MoE_IDS
from src.utils.logging import setup_logging, MODELS_DIR

log = logging.getLogger(__name__)

INFERENCE_DIR = Path("results/inference")

FOLD_TRAIN = {
    "lycos_ids2017": ["cic_ids2018",   "unsw_nb15",   "ton_iot"],
    "cic_ids2018":   ["lycos_ids2017", "unsw_nb15",   "ton_iot"],
    "unsw_nb15":     ["lycos_ids2017", "cic_ids2018", "ton_iot"],
    "ton_iot":       ["lycos_ids2017", "cic_ids2018", "unsw_nb15"],
}

# attack_class vocabulary (int8 encoding)
AC_VOCAB = ["Benign", "Reconnaissance", "DoS_DDoS",
            "Injection_Exploit", "BruteForce", "Botnet_C2"]
AC_MAP   = {c: i for i, c in enumerate(AC_VOCAB)}

# ── graph cache (one graph at a time) ────────────────────────────────────────
# Cleared explicitly between test-fold groups.
_GRAPH_CACHE: dict = {}   # {(fold, tier, dev, within): Data}

def _evict_graph_cache():
    """Free all cached graphs and force GC."""
    _GRAPH_CACHE.clear()
    gc.collect()


def _load_cached_graph(fold: str, tier: str, dev: bool, within: bool = False):
    """
    Load a test graph, caching it for re-use across checkpoints on the same fold.
    Immediately deletes ip_to_idx and feature_cols (unneeded, can be huge).
    """
    key = (fold, tier, dev, within)
    if key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]

    # Different graph needed: evict everything first
    _evict_graph_cache()

    log.info(f"  Loading {fold} tier={tier} dev={dev} …")
    g = load_graph(fold, tier=tier, dev=dev)

    # Free attrs we never use for inference
    for attr in ("ip_to_idx", "feature_cols"):
        if hasattr(g, attr):
            delattr(g, attr)

    if within:
        sp  = load_split(fold)
        idx = torch.as_tensor(sp["val"], dtype=torch.long)
        g2  = copy.copy(g)
        g2.edge_index      = g.edge_index[:, idx]
        g2.edge_attr       = g.edge_attr[idx]
        g2.edge_attr_q     = g.edge_attr_q[idx]
        g2.edge_time       = g.edge_time[idx]
        g2.edge_label      = g.edge_label[idx]
        g2.edge_label_type = [g.edge_label_type[i] for i in sp["val"]]
        del g
        g = g2

    _GRAPH_CACHE[key] = g
    log.info(f"    edges={g.edge_label.shape[0]:,}  nodes={g.x.shape[0]:,}")
    return g


# ── attack-class encoding / decoding ─────────────────────────────────────────

def _encode_ac(ac_list: list) -> np.ndarray:
    """Convert list of class-name strings → int8 array."""
    return np.array([AC_MAP.get(c, -1) for c in ac_list], dtype=np.int8)


def decode_ac(ac_ids: np.ndarray, vocab=None) -> list:
    """Decode int8 array → list of strings (used in run_phase11)."""
    v = vocab if vocab is not None else AC_VOCAB
    return [v[i] if 0 <= i < len(v) else "Unknown" for i in ac_ids]


# ── filename parsing ──────────────────────────────────────────────────────────

def parse_ckpt(stem: str):
    if re.search(r'E5\.2_spec\d', stem):
        return {"role": "skip"}
    if re.match(r'E6\.\d+_\w+_if_seed', stem):
        return {"role": "skip"}

    if re.match(r'E5\.2_gate_seed', stem):
        m = re.search(r'_seed(\d+)_test(.+)$', stem)
        if m:
            return {"method": "E5.2_source_gated", "seed": int(m.group(1)),
                    "test_fold": m.group(2), "role": "gate_trigger"}

    if re.search(r'E6\.\d+_\w+_encoder_seed', stem):
        m = re.search(r'_seed(\d+)_test(.+)$', stem)
        if m:
            prefix = stem[:stem.index("_encoder_seed")]
            return {"method": prefix, "seed": int(m.group(1)),
                    "test_fold": m.group(2), "role": "anomaly_encoder"}

    m = re.search(r'_seed(\d+)_test(.+)$', stem)
    if not m:
        return None
    return {"method": stem[:m.start()], "seed": int(m.group(1)),
            "test_fold": m.group(2), "role": "main"}


# ── architecture detection ────────────────────────────────────────────────────

def detect_arch(sd):
    keys = set(sd.keys())
    if any(k.startswith("experts.") for k in keys): return "moe"
    # TS_GIB uses ctx_enc (not edge_enc) — must check before generic "gib"
    if "ctx_enc.0.weight" in keys:                  return "ts_gib"
    if "to_dist.weight" in keys:                    return "gib"
    if "encoder.edge_enc.0.weight" in keys:         return "dann"
    # TemporalEdgeSAGE has edge_enc + norm1 (LayerNorm); plain SAGE does not
    if "edge_enc.0.weight" in keys and "norm1.weight" in keys: return "ts_sage"
    if "edge_enc.0.weight" in keys:                 return "sage"
    return "unknown"

def _get_edge_in(sd, arch):
    if arch == "ts_gib":       return sd["ctx_enc.0.weight"].shape[1]   # ctx_edge_in (always 1)
    if arch == "ts_sage":      return sd["edge_enc.0.weight"].shape[1]
    if arch in ("sage","gib"): return sd["edge_enc.0.weight"].shape[1]
    if arch == "dann":         return sd["encoder.edge_enc.0.weight"].shape[1]
    if arch == "moe":          return sd["experts.0.edge_enc.0.weight"].shape[1]
    return 1

def _get_hidden(sd, arch):
    if arch == "ts_gib":       return sd["ctx_enc.0.weight"].shape[0]
    if arch == "ts_sage":      return sd["edge_enc.0.weight"].shape[0]
    if arch in ("sage","gib"): return sd["edge_enc.0.weight"].shape[0]
    if arch == "dann":         return sd["encoder.edge_enc.0.weight"].shape[0]
    if arch == "moe":          return sd["experts.0.edge_enc.0.weight"].shape[0]
    return 128

def _get_node_in(sd, arch, hidden):
    if arch in ("ts_gib", "ts_sage"):
                               return sd["conv1.lin_l.weight"].shape[1] - hidden
    if arch in ("sage","gib"): return sd["conv1.lin_l.weight"].shape[1] - hidden
    if arch == "dann":         return sd["encoder.conv1.lin_l.weight"].shape[1] - hidden
    if arch == "moe":          return sd["experts.0.conv1.lin_l.weight"].shape[1] - hidden
    return 8


# ── memory-efficient inference ────────────────────────────────────────────────

@torch.no_grad()
def _infer_local(model_fn, x, ei, ea, device, bs=20_000):
    """
    Local-subgraph batched inference with a PRE-ALLOCATED assoc buffer.

    Previously passing full x to model() caused scatter(dim_size=N_total) to
    allocate a [N_total × H] tensor on EVERY batch – fatal for large graphs.
    Here we remap each batch to its local nodes, so scatter only sees ~2×bs nodes.
    The assoc buffer is allocated once (N_total × 8 bytes) and reused.
    """
    N = x.size(0)
    E = ei.shape[1]
    assoc = torch.empty(N, dtype=torch.long, device=device)
    parts = []
    for s in range(0, E, bs):
        ei_b   = ei[:, s:s+bs]
        ea_b   = ea[s:s+bs]
        n_ids  = ei_b.reshape(-1).unique()
        assoc.fill_(-1)
        assoc[n_ids] = torch.arange(n_ids.size(0), device=device)
        logits = model_fn(x[n_ids], assoc[ei_b], ea_b)
        parts.append(F.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        del n_ids, logits
    del assoc
    return np.concatenate(parts).astype(np.float32)


@torch.no_grad()
def _anomaly_scores_batched(encoder, iforest, x, ei, ea, device, bs=4096):
    """
    Embed → IF-score in small batches, never holding full embedding matrix.
    Peak extra RAM = bs × embed_dim × 4 bytes ≈ 6 MB at bs=4096, dim=384.
    Same local-subgraph approach so assoc is pre-allocated once.
    """
    N = x.size(0)
    E = ei.shape[1]
    assoc = torch.empty(N, dtype=torch.long, device=device)
    score_parts = []
    for s in range(0, E, bs):
        ei_b  = ei[:, s:s+bs]
        ea_b  = ea[s:s+bs]
        n_ids = ei_b.reshape(-1).unique()
        assoc.fill_(-1)
        assoc[n_ids] = torch.arange(n_ids.size(0), device=device)
        emb   = encoder.embed(x[n_ids], assoc[ei_b], ea_b).cpu().numpy()
        score_parts.append(-iforest.score_samples(emb).astype(np.float32))
        del n_ids, emb
    del assoc
    return np.concatenate(score_parts)


# ── graph feature helpers ─────────────────────────────────────────────────────

def _struct_only(g):
    g2 = copy.copy(g)
    E  = g2.edge_attr.shape[0]
    g2.edge_attr   = torch.ones(E, 1)
    g2.edge_attr_q = torch.ones(E, 1)
    return g2


def _pad_or_crop(g, target_dim):
    g2 = copy.copy(g)
    d  = g2.edge_attr.shape[1]
    if d < target_dim:
        pad = torch.zeros(g2.edge_attr.shape[0], target_dim - d)
        g2.edge_attr   = torch.cat([g2.edge_attr,   pad], dim=1)
        g2.edge_attr_q = torch.cat([g2.edge_attr_q, pad], dim=1)
    elif d > target_dim:
        g2.edge_attr   = g2.edge_attr[:, :target_dim]
        g2.edge_attr_q = g2.edge_attr_q[:, :target_dim]
    return g2


def _get_graph_tensors(g, edge_in, method, dev, device):
    """
    Return (x, ei, ea, labels_np, ac_ids) on device, with appropriate features.
    labels_np and ac_ids are numpy arrays (copied, not aliasing graph tensors).
    """
    within = "B3_within" in method

    if edge_in == 1:
        g = _load_cached_graph(g.fold_ if hasattr(g, 'fold_') else _current_fold,
                               "B", dev, within)
        g = _struct_only(g)
    elif edge_in == 4:
        suffix = "_dev" if dev else "_full"
        tier   = "A"
        for kind in ("lqe", "lze"):
            if kind.lower() in method.lower():
                cache = PROCESSED_DIR / f"{_current_fold}_tier{tier}_{kind}{suffix}.pt"
                if cache.exists():
                    g = torch.load(cache, weights_only=False)
                    break
    elif edge_in > 4:
        g = _pad_or_crop(g, edge_in)

    x  = g.x.to(device)
    ei = g.edge_index.to(device)
    ea = g.edge_attr_q.to(device)
    labels_np = g.edge_label.numpy().copy()
    ac_ids    = _encode_ac(g.edge_label_type)
    return x, ei, ea, labels_np, ac_ids


# ── temporal processors ───────────────────────────────────────────────────────

def _anom_scores_for_temporal(test_fold: str, seed: int, dev: bool, device: str):
    """
    Load E6.2 MSA encoder + IsolationForest and score test-graph edges.
    Returns float32 [E_test] array, or None if artifacts are missing.
    Used for E12.2 / E13.1 checkpoints that append anomaly score to q_ea.
    """
    enc_path = MODELS_DIR / f"E6.2_msa_encoder_seed{seed}_test{test_fold}.pt"
    if_path  = MODELS_DIR / f"E6.2_msa_if_seed{seed}_test{test_fold}.pt"
    if not enc_path.exists() or not if_path.exists():
        log.warning(f"  E6.2 artifacts missing for seed={seed} test={test_fold}")
        return None
    try:
        enc_sd  = torch.load(enc_path, weights_only=True)
        enc_ein = enc_sd["edge_enc.0.weight"].shape[1]
        enc_h   = enc_sd["edge_enc.0.weight"].shape[0]
        enc_nin = enc_sd["conv1.lin_l.weight"].shape[1] - enc_h
        encoder = EdgeAwareSAGE(node_in=enc_nin, edge_in=enc_ein, hidden=enc_h)
        encoder.load_state_dict(enc_sd); del enc_sd
        encoder.eval().to(device)
        iforest = torch.load(if_path, weights_only=False)
        g_a   = _load_cached_graph(test_fold, "A", dev)
        x_a   = g_a.x.to(device)
        ei_a  = g_a.edge_index.to(device)
        ea_a  = g_a.edge_attr_q.to(device)
        scores = _anomaly_scores_batched(encoder, iforest, x_a, ei_a, ea_a, device)
        del encoder, iforest, x_a, ei_a, ea_a
        return scores.astype(np.float32)
    except Exception as exc:
        log.warning(f"  Failed to compute anomaly scores: {exc}")
        return None


@torch.no_grad()
def process_temporal(ckpt_path, info, dev, device):
    """
    Score extraction for temporal models:
      ts_gib  — TS_GIB (E12.x, E13.1-2) with optional variational bottleneck
      ts_sage — TemporalEdgeSAGE (E8.x)

    Builds temporal subgraphs from the test graph using structure-only edge
    features (constant 1.0), matching the training convention of these models.
    For checkpoints with a separate query encoder (E12.2 / E13.1), loads E6.2
    anomaly scores and appends them to the query-edge feature tensor.
    """
    from run_phase12 import (
        _graph_arrays, _delta_us,
        DELTA_SECS, BATCH_SIZE, MAX_EVAL_EDGES, MAX_SUB_EDGES, NODE_FEAT_DIM, N_JOBS,
    )
    from src.data.temporal_subgraph import batch_build_subgraphs
    from src.models.temporal_gnn import TS_GIB, TemporalEdgeSAGE
    from torch_geometric.data import Batch as _Batch
    from sklearn.model_selection import train_test_split as _tts

    test_fold = info["test_fold"]
    seed      = info["seed"]

    state = torch.load(ckpt_path, weights_only=True)
    keys  = set(state.keys())
    arch  = detect_arch(state)

    # ── Reconstruct model ───────────────────────────────────────────────────
    if arch == "ts_gib":
        hidden    = state["ctx_enc.0.weight"].shape[0]
        node_in   = state["conv1.lin_l.weight"].shape[1] - hidden
        has_q_enc = any(k.startswith("q_enc.") for k in keys)
        q_edge_in = state["q_enc.0.weight"].shape[1] if has_q_enc else 1
        use_bn    = "to_dist.weight" in keys
        dom_w     = state.get("domain_head.2.weight")
        num_doms  = dom_w.shape[0] if dom_w is not None else 0
        model     = TS_GIB(node_in=node_in, ctx_edge_in=1, q_edge_in=q_edge_in,
                           hidden=hidden, use_bottleneck=use_bn, num_domains=num_doms)
    else:  # ts_sage — TemporalEdgeSAGE
        edge_in = state["edge_enc.0.weight"].shape[1]
        hidden  = state["edge_enc.0.weight"].shape[0]
        node_in = state["conv1.lin_l.weight"].shape[1] - hidden
        model   = TemporalEdgeSAGE(node_in=node_in, edge_in=edge_in, hidden=hidden)
        q_edge_in = edge_in

    model.load_state_dict(state); del state
    model.eval().to(device)

    # ── Load test graph (structure-only) ────────────────────────────────────
    g         = _load_cached_graph(test_fold, "B", dev)
    E_all     = g.edge_attr.shape[0]
    src_np, dst_np, time_np = _graph_arrays(g)
    labels_np = g.edge_label.numpy().copy()
    ac_ids    = _encode_ac(g.edge_label_type)

    # ── Stratified test cap ─────────────────────────────────────────────────
    n_total = len(labels_np)
    if n_total > MAX_EVAL_EDGES:
        _, eval_idx = _tts(np.arange(n_total),
                           test_size=MAX_EVAL_EDGES / n_total,
                           random_state=42, stratify=labels_np)
        eval_idx  = np.sort(np.array(eval_idx, dtype=np.int64))
        labels_np = labels_np[eval_idx]
        ac_ids    = ac_ids[eval_idx]
        log.info(f"  Test capped: {n_total:,} → {len(eval_idx):,}")
    else:
        eval_idx = np.arange(n_total, dtype=np.int64)

    # ── Anomaly scores for E12.2 / E13.1 (q_edge_in > 1) ───────────────────
    anom_np = None
    if arch == "ts_gib" and q_edge_in == 2:
        anom_np = _anom_scores_for_temporal(test_fold, seed, dev, device)
        if anom_np is None:
            log.warning(f"  Falling back to zeros for anomaly q_ea")
            anom_np = np.zeros(E_all, dtype=np.float32)

    # ── Pre-extract all subgraphs ────────────────────────────────────────────
    du = _delta_us(DELTA_SECS)
    log.info(f"  Extracting {len(eval_idx):,} temporal subgraphs (n_jobs={N_JOBS}) …")
    t0 = time.time()
    all_data = batch_build_subgraphs(
        src_np, dst_np, time_np,
        src_np[eval_idx], dst_np[eval_idx], time_np[eval_idx],
        delta_us=du, max_edges=MAX_SUB_EDGES,
        node_feat_dim=NODE_FEAT_DIM, seed=0, n_jobs=N_JOBS,
    )
    log.info(f"  Subgraph extraction done in {time.time()-t0:.1f}s")

    # ── Batched inference ────────────────────────────────────────────────────
    all_scores = []
    for start in range(0, len(eval_idx), BATCH_SIZE):
        ids_b = eval_idx[start:start + BATCH_SIZE]
        B_    = len(ids_b)

        # Query-edge features
        if anom_np is not None:
            a    = torch.as_tensor(anom_np[ids_b], dtype=torch.float32, device=device)
            q_ea = torch.stack([torch.ones(B_, device=device), a], dim=-1)
        else:
            q_ea = torch.ones(B_, q_edge_in, device=device)

        # Batch subgraphs and recover global src/dst indices
        dl    = all_data[start:start + BATCH_SIZE]
        batch = _Batch.from_data_list([d.to(device) for d in dl])
        ptr   = batch.ptr.to(device)
        u_globals = torch.tensor([d.query_u for d in dl],
                                  dtype=torch.long, device=device) + ptr[:-1]
        v_globals = torch.tensor([d.query_v for d in dl],
                                  dtype=torch.long, device=device) + ptr[:-1]

        out    = model(batch.x, batch.edge_index, batch.edge_attr,
                       u_globals, v_globals, q_ea)
        logits = out[0] if isinstance(out, tuple) else out
        all_scores.append(F.softmax(logits, dim=-1)[:, 1].cpu().numpy())

    del model, all_data
    return np.concatenate(all_scores).astype(np.float32), labels_np, ac_ids


# ── processors ───────────────────────────────────────────────────────────────

_current_fold = None   # set in main loop so helpers can read it


def _graph_for_fold(test_fold, edge_in, method, dev):
    """Return the appropriate graph (possibly struct-only / LQE / tier-B padded)."""
    within = "B3_within" in method

    if edge_in == 1:
        g = _load_cached_graph(test_fold, "B", dev, within)
        return _struct_only(g)

    if edge_in == 4:
        suffix = "_dev" if dev else "_full"
        for kind in ("lqe", "lze"):
            if kind.lower() in method.lower():
                cache = PROCESSED_DIR / f"{test_fold}_tierA_{kind}{suffix}.pt"
                if cache.exists():
                    return torch.load(cache, weights_only=False)
        return _load_cached_graph(test_fold, "A", dev, within)

    # tier B padded/cropped
    g = _load_cached_graph(test_fold, "B", dev, within)
    return _pad_or_crop(g, edge_in)


def process_main(ckpt_path, info, dev, device):
    sd     = torch.load(ckpt_path, weights_only=True)
    arch   = detect_arch(sd)
    if arch == "unknown":
        log.warning(f"  Unknown arch: {ckpt_path.name}")
        del sd; return None

    # Temporal models need subgraph extraction — delegate to dedicated handler
    if arch in ("ts_gib", "ts_sage"):
        del sd
        return process_temporal(ckpt_path, info, dev, device)

    method    = info["method"]
    test_fold = info["test_fold"]
    edge_in   = _get_edge_in(sd, arch)
    hidden    = _get_hidden(sd, arch)
    node_in   = _get_node_in(sd, arch, hidden)

    if edge_in == 2 and "hybrid" in method.lower():
        return process_e64_hybrid(ckpt_path, sd, info, dev, device,
                                  node_in, edge_in, hidden)

    g   = _graph_for_fold(test_fold, edge_in, method, dev)
    x   = g.x.to(device)
    ei  = g.edge_index.to(device)
    ea  = g.edge_attr_q.to(device)
    labels_np = g.edge_label.numpy().copy()
    ac_ids    = _encode_ac(g.edge_label_type)
    # g stays in the cache; x/ei/ea are device views

    scores = None
    try:
        if arch == "sage":
            model = EdgeAwareSAGE(node_in=node_in, edge_in=edge_in,
                                  hidden=hidden, num_classes=2)
            model.load_state_dict(sd); del sd
            model.eval().to(device)
            scores = _infer_local(model, x, ei, ea, device)

        elif arch == "gib":
            bn_dim = sd["to_dist.weight"].shape[0] // 2
            model  = GIB_EGraphSAGE(node_in=node_in, edge_in=edge_in,
                                     hidden=hidden, bottleneck_dim=bn_dim)
            model.load_state_dict(sd); del sd
            model.eval().to(device)
            scores = _infer_local(model, x, ei, ea, device)

        elif arch == "dann":
            dom_w = sd.get("domain_head.2.weight")
            nd    = dom_w.shape[0] if dom_w is not None else 3
            try:
                from run_phase4 import DANN_EGS
                model = DANN_EGS(node_in=node_in, edge_in=edge_in,
                                 hidden=hidden, num_domains=nd)
                model.load_state_dict(sd); del sd
            except RuntimeError:
                from run_phase5 import CDAN_EGS
                model = CDAN_EGS(node_in=node_in, edge_in=edge_in,
                                 hidden=hidden, num_domains=nd)
                model.load_state_dict(sd); del sd
            model.eval().to(device)
            scores = _infer_local(
                lambda xb, eib, eab: model.predict(xb, eib, eab),
                x, ei, ea, device)

        elif arch == "moe":
            K     = max(int(k.split(".")[1]) for k in sd if k.startswith("experts."))
            model = MoE_IDS(node_in=node_in, edge_in=edge_in, hidden=hidden, K=K)
            model.load_state_dict(sd); del sd
            model.eval().to(device)
            scores = _infer_local(model, x, ei, ea, device)

        else:
            del sd; return None
    finally:
        del x, ei, ea
        try: del model
        except NameError: pass

    return scores, labels_np, ac_ids


def process_e64_hybrid(ckpt_path, sd, info, dev, device, node_in, edge_in, hidden):
    test_fold = info["test_fold"]
    seed      = info["seed"]
    method    = info["method"]

    base_m = re.search(r'E6\.(\d+)', method)
    if not base_m:
        del sd; return None
    base_prefix = "E6.2_msa" if base_m.group(1) == "2" else "E6.1_anomal_e"
    enc_path = MODELS_DIR / f"{base_prefix}_encoder_seed{seed}_test{test_fold}.pt"
    if_path  = MODELS_DIR / f"{base_prefix}_if_seed{seed}_test{test_fold}.pt"
    if not enc_path.exists() or not if_path.exists():
        log.warning(f"  Missing base encoder/IF for {method}"); del sd; return None

    enc_sd  = torch.load(enc_path, weights_only=True)
    enc_ein = enc_sd["edge_enc.0.weight"].shape[1]
    enc_h   = enc_sd["edge_enc.0.weight"].shape[0]
    enc_nin = enc_sd["conv1.lin_l.weight"].shape[1] - enc_h
    encoder = EdgeAwareSAGE(node_in=enc_nin, edge_in=enc_ein, hidden=enc_h)
    encoder.load_state_dict(enc_sd); del enc_sd
    encoder.eval().to(device)
    iforest = torch.load(if_path, weights_only=False)

    g_a = _load_cached_graph(test_fold, "A", dev)
    x_a = g_a.x.to(device)
    ei_a = g_a.edge_index.to(device)
    ea_a = g_a.edge_attr_q.to(device)

    anom = _anomaly_scores_batched(encoder, iforest, x_a, ei_a, ea_a, device)
    del encoder, iforest, x_a, ei_a, ea_a

    sc   = torch.as_tensor(anom, dtype=torch.float32).unsqueeze(1); del anom
    ea_h = torch.cat([torch.ones_like(sc), sc], dim=1); del sc
    # Build a lightweight wrapper (avoid copying the full graph)
    class _FakeGraph:
        pass
    g_h = _FakeGraph()
    g_h.x          = g_a.x
    g_h.edge_index  = g_a.edge_index
    g_h.edge_attr   = ea_h
    g_h.edge_attr_q = ea_h

    model = EdgeAwareSAGE(node_in=node_in, edge_in=edge_in, hidden=hidden)
    model.load_state_dict(sd); del sd
    model.eval().to(device)
    x  = g_h.x.to(device)
    ei = g_h.edge_index.to(device)
    ea = g_h.edge_attr_q.to(device)
    scores = _infer_local(model, x, ei, ea, device)
    del model, x, ei, ea, ea_h

    labels_np = g_a.edge_label.numpy().copy()
    ac_ids    = _encode_ac(g_a.edge_label_type)
    return scores, labels_np, ac_ids


def process_anomaly_encoder(ckpt_path, info, dev, device):
    test_fold = info["test_fold"]
    if_path   = ckpt_path.parent / ckpt_path.name.replace("_encoder_", "_if_")
    if not if_path.exists():
        log.warning(f"  IF not found: {if_path}"); return None

    sd      = torch.load(ckpt_path, weights_only=True)
    edge_in = sd["edge_enc.0.weight"].shape[1]
    hidden  = sd["edge_enc.0.weight"].shape[0]
    node_in = sd["conv1.lin_l.weight"].shape[1] - hidden
    encoder = EdgeAwareSAGE(node_in=node_in, edge_in=edge_in, hidden=hidden)
    encoder.load_state_dict(sd); del sd
    encoder.eval().to(device)
    iforest = torch.load(if_path, weights_only=False)

    g  = _load_cached_graph(test_fold, "A", dev)
    x  = g.x.to(device)
    ei = g.edge_index.to(device)
    ea = g.edge_attr_q.to(device)

    scores = _anomaly_scores_batched(encoder, iforest, x, ei, ea, device)
    del encoder, iforest, x, ei, ea

    labels_np = g.edge_label.numpy().copy()
    ac_ids    = _encode_ac(g.edge_label_type)
    return scores, labels_np, ac_ids


def process_gate(ckpt_path, info, dev, device):
    test_fold   = info["test_fold"]
    seed        = info["seed"]
    train_dsets = FOLD_TRAIN[test_fold]
    n_src       = len(train_dsets)

    sd_gate  = torch.load(ckpt_path, weights_only=True)
    in_feat  = sd_gate["net.0.weight"].shape[1]
    hidden_g = sd_gate["net.0.weight"].shape[0]
    from run_phase5 import GateNetwork
    gate = GateNetwork(in_features=in_feat, hidden=hidden_g, num_sources=n_src)
    gate.load_state_dict(sd_gate); del sd_gate
    gate.eval().to(device)

    from run_phase4 import DANN_EGS
    specialists = []
    for src_d in range(n_src):
        spec_path = ckpt_path.parent / ckpt_path.name.replace("_gate_", f"_spec{src_d}_")
        if not spec_path.exists():
            log.warning(f"  Specialist {src_d} missing: {spec_path}"); return None
        sd_s  = torch.load(spec_path, weights_only=True)
        ein   = sd_s["encoder.edge_enc.0.weight"].shape[1]
        h     = sd_s["encoder.edge_enc.0.weight"].shape[0]
        nin   = sd_s["encoder.conv1.lin_l.weight"].shape[1] - h
        dom_w = sd_s.get("domain_head.2.weight")
        nd    = dom_w.shape[0] if dom_w is not None else n_src
        spec  = DANN_EGS(node_in=nin, edge_in=ein, hidden=h, num_domains=nd)
        spec.load_state_dict(sd_s); del sd_s
        specialists.append(spec)

    g_b    = _load_cached_graph(test_fold, "B", dev)
    g_s    = _struct_only(g_b)
    g_a    = _load_cached_graph(test_fold, "A", dev)
    ea_q_a = quantile_encode(g_a.edge_attr)
    E      = g_s.edge_label.shape[0]

    ea_a_dev = ea_q_a.to(device)
    gate_parts = []
    with torch.no_grad():
        for s in range(0, E, 50_000):
            gate_parts.append(gate(ea_a_dev[s:s+50_000]).cpu())
    w = torch.cat(gate_parts, dim=0); del gate_parts, ea_a_dev, gate

    x_t  = g_s.x.to(device)
    ei_t = g_s.edge_index.to(device)
    ea_t = g_s.edge_attr_q.to(device)
    N    = x_t.size(0)
    assoc = torch.empty(N, dtype=torch.long, device=device)

    combined = torch.zeros(E, 2)
    for src_d, spec in enumerate(specialists):
        spec.eval().to(device)
        parts = []
        with torch.no_grad():
            for s in range(0, E, 20_000):
                ei_b  = ei_t[:, s:s+20_000]
                ea_b  = ea_t[s:s+20_000]
                n_ids = ei_b.reshape(-1).unique()
                assoc.fill_(-1)
                assoc[n_ids] = torch.arange(n_ids.size(0), device=device)
                parts.append(spec.predict(x_t[n_ids], assoc[ei_b], ea_b).cpu())
                del n_ids, ei_b, ea_b
        logits = torch.cat(parts, dim=0)
        combined += w[:, src_d:src_d+1] * logits
        del parts, logits
        spec.cpu()

    del assoc, x_t, ei_t, ea_t, w, specialists

    scores    = F.softmax(combined, dim=-1)[:, 1].numpy().astype(np.float32)
    del combined
    labels_np = g_s.edge_label.numpy().copy()
    ac_ids    = _encode_ac(g_s.edge_label_type)
    return scores, labels_np, ac_ids


# ── atomic save ───────────────────────────────────────────────────────────────

def _atomic_save(obj, out_path):
    tmp = out_path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.rename(out_path)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev",       action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--method",    default=None)
    args = parser.parse_args()

    setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    # ── collect and sort by test_fold ────────────────────────────────────────
    # Grouping by test_fold means each test graph is loaded ONCE for all its
    # checkpoints, then freed before the next fold's graph is loaded.
    all_ckpts = sorted(MODELS_DIR.glob("*.pt"))
    log.info(f"Found {len(all_ckpts)} checkpoints")

    # Parse all names first (fast, no disk I/O)
    parsed = []
    for p in all_ckpts:
        info = parse_ckpt(p.stem)
        if info is None or info["role"] == "skip":
            continue
        parsed.append((p, info))

    # Sort: test_fold first, then original filename order within fold
    FOLD_ORDER = {"lycos_ids2017": 0, "cic_ids2018": 1, "unsw_nb15": 2, "ton_iot": 3}
    parsed.sort(key=lambda t: (FOLD_ORDER.get(t[1].get("test_fold",""), 99), t[0].name))

    done = skip = fail = 0
    current_fold_key = None

    for ckpt_path, info in parsed:
        if args.method and args.method not in ckpt_path.stem:
            continue

        out_path = INFERENCE_DIR / f"{ckpt_path.stem}.pt"
        if out_path.exists() and not args.overwrite:
            log.info(f"  Skip (exists): {ckpt_path.stem}")
            done += 1
            continue

        # When we move to a new fold, evict the cached graph for the old fold
        fold_key = info.get("test_fold", "")
        if fold_key != current_fold_key:
            log.info(f"\n── Fold: {fold_key} ──")
            _evict_graph_cache()
            current_fold_key = fold_key

        log.info(f"Processing {ckpt_path.stem} …")
        result = None
        scores = labels_np = ac_ids = None
        try:
            role = info["role"]
            if role == "anomaly_encoder":
                result = process_anomaly_encoder(ckpt_path, info, args.dev, device)
            elif role == "gate_trigger":
                result = process_gate(ckpt_path, info, args.dev, device)
            else:
                result = process_main(ckpt_path, info, args.dev, device)

            if result is None:
                log.warning(f"  No result: {ckpt_path.stem}"); skip += 1; continue

            scores, labels_np, ac_ids = result
            _atomic_save({
                "scores":            scores,
                "labels":            labels_np.astype(np.int8),
                "attack_class_ids":  ac_ids,
                "attack_class_vocab": AC_VOCAB,
                "method":            info["method"],
                "seed":              info["seed"],
                "test_fold":         info["test_fold"],
            }, out_path)
            log.info(f"  Saved  E={len(scores):,}  "
                     f"atk={labels_np.mean():.3f}  "
                     f"scores=[{scores.min():.3f},{scores.max():.3f}]")
            done += 1

        except MemoryError:
            log.error(f"  OOM on {ckpt_path.stem} — skipping. "
                      "Consider --dev for smaller graphs.")
            fail += 1
        except Exception as exc:
            log.error(f"  FAILED {ckpt_path.stem}: {exc}", exc_info=True)
            fail += 1
        finally:
            del result, scores, labels_np, ac_ids
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    _evict_graph_cache()
    log.info(f"\nDone={done}  Skip={skip}  Fail={fail}")


if __name__ == "__main__":
    main()
