#!/usr/bin/env python3
"""
Phase 11 A.1: Score extraction across all checkpoints.

Iterates results/models/*.pt, loads each model, runs inference on the
corresponding test fold, saves per-edge {scores, labels, attack_classes}
to results/inference/<basename>.pt.

Three loader families:
  1. EdgeAwareSAGE family  — B3/B4/B5, E1-E5, E7-E10, P*
  2. Source-conditional gating (E5.2) — gate + 3 specialists
  3. Anomal-E (E6.1, E6.2)  — EdgeAwareSAGE encoder + IsolationForest

Memory budget: designed for ≤26 GB RAM (no full embedding matrix in memory).
  Anomaly scoring is done in small batches; models and graphs are explicitly
  freed after each checkpoint; saves use atomic rename so partial files are
  never left behind on crash.

Usage (on remote server):
    python scripts/extract_scores.py [--dev] [--overwrite]
    python scripts/extract_scores.py --method E1.A  # single method filter
"""

import argparse
import copy
import gc
import logging
import re
import sys
import tempfile
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

# ── filename parsing ──────────────────────────────────────────────────────────

def parse_ckpt(stem: str):
    """
    Returns dict with (method, seed, test_fold, role).
    role: 'main' | 'skip' | 'anomaly_encoder' | 'gate_trigger'
    Returns None if pattern doesn't match.
    """
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

def detect_arch(sd: dict) -> str:
    keys = set(sd.keys())
    if any(k.startswith("experts.") for k in keys):
        return "moe"
    if "to_dist.weight" in keys:
        return "gib"
    if "encoder.edge_enc.0.weight" in keys:
        return "dann"
    if "edge_enc.0.weight" in keys:
        return "sage"
    return "unknown"


def _get_edge_in(sd, arch):
    if arch in ("sage", "gib"):  return sd["edge_enc.0.weight"].shape[1]
    if arch == "dann":           return sd["encoder.edge_enc.0.weight"].shape[1]
    if arch == "moe":            return sd["experts.0.edge_enc.0.weight"].shape[1]
    return 1

def _get_hidden(sd, arch):
    if arch in ("sage", "gib"):  return sd["edge_enc.0.weight"].shape[0]
    if arch == "dann":           return sd["encoder.edge_enc.0.weight"].shape[0]
    if arch == "moe":            return sd["experts.0.edge_enc.0.weight"].shape[0]
    return 128

def _get_node_in(sd, arch, hidden):
    if arch in ("sage", "gib"):  return sd["conv1.lin_l.weight"].shape[1] - hidden
    if arch == "dann":           return sd["encoder.conv1.lin_l.weight"].shape[1] - hidden
    if arch == "moe":            return sd["experts.0.conv1.lin_l.weight"].shape[1] - hidden
    return 8


# ── test graph loading ────────────────────────────────────────────────────────

def _struct_only(g):
    g2 = copy.copy(g)
    E  = g2.edge_attr.shape[0]
    g2.edge_attr   = torch.ones(E, 1)
    g2.edge_attr_q = torch.ones(E, 1)
    return g2


def _pad_or_crop(g, target_dim: int):
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


def _index_graph(g, idx_list):
    idx = torch.as_tensor(idx_list, dtype=torch.long)
    g2  = copy.copy(g)
    g2.edge_index      = g.edge_index[:, idx]
    g2.edge_attr       = g.edge_attr[idx]
    g2.edge_attr_q     = g.edge_attr_q[idx]
    g2.edge_time       = g.edge_time[idx]
    g2.edge_label      = g.edge_label[idx]
    g2.edge_label_type = [g.edge_label_type[i] for i in idx_list]
    return g2


def load_test_graph(test_fold: str, edge_in: int, method: str,
                    dev: bool, within: bool = False):
    if edge_in == 1:
        g = load_graph(test_fold, tier="B", dev=dev)
        if within:
            g = _index_graph(g, load_split(test_fold)["val"])
        return _struct_only(g)

    if edge_in == 4:
        suffix = "_dev" if dev else "_full"
        for kind in ("lqe", "lze"):
            cache = PROCESSED_DIR / f"{test_fold}_tierA_{kind}{suffix}.pt"
            if cache.exists() and kind.lower() in method.lower():
                g = torch.load(cache, weights_only=False)
                if within:
                    g = _index_graph(g, load_split(test_fold)["val"])
                return g
        g = load_graph(test_fold, tier="A", dev=dev)
        if within:
            g = _index_graph(g, load_split(test_fold)["val"])
        return g

    g = load_graph(test_fold, tier="B", dev=dev)
    if within:
        g = _index_graph(g, load_split(test_fold)["val"])
    return _pad_or_crop(g, edge_in)


# ── inference helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def _infer_sage(model, graph, device, bs=50_000):
    model.eval().to(device)
    x  = graph.x.to(device)
    ei = graph.edge_index.to(device)
    ea = graph.edge_attr_q.to(device)
    E  = ei.shape[1]
    parts = []
    for s in range(0, E, bs):
        logits = model(x, ei[:, s:s+bs], ea[s:s+bs])
        parts.append(F.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    del x, ei, ea
    return np.concatenate(parts).astype(np.float32)


@torch.no_grad()
def _infer_dann(model, graph, device, bs=50_000):
    model.eval().to(device)
    x  = graph.x.to(device)
    ei = graph.edge_index.to(device)
    ea = graph.edge_attr_q.to(device)
    E  = ei.shape[1]
    parts = []
    for s in range(0, E, bs):
        logits = model.predict(x, ei[:, s:s+bs], ea[s:s+bs])
        parts.append(F.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    del x, ei, ea
    return np.concatenate(parts).astype(np.float32)


@torch.no_grad()
def _anomaly_scores_batched(encoder, iforest, graph, device, bs=4096):
    """
    Score edges via IsolationForest WITHOUT holding the full embedding matrix.
    Peak extra RAM = bs × embed_dim × 4 bytes  (≈6 MB at bs=4096, dim=384).
    """
    from run_phase6 import _to_local_graph
    encoder.eval().to(device)
    x  = graph.x.to(device)
    ei = graph.edge_index.to(device)
    ea = graph.edge_attr_q.to(device)
    E  = ei.shape[1]
    score_parts = []
    for s in range(0, E, bs):
        ids = np.arange(s, min(s + bs, E))
        x_b, ei_b, ea_b = _to_local_graph(x, ei, ea, ids, device)
        emb = encoder.embed(x_b, ei_b, ea_b).cpu().numpy()
        score_parts.append(-iforest.score_samples(emb).astype(np.float32))
        del x_b, ei_b, ea_b, emb          # free immediately each batch
    del x, ei, ea
    return np.concatenate(score_parts)


# ── processors ───────────────────────────────────────────────────────────────

def process_main(ckpt_path: Path, info: dict, dev: bool, device: str):
    sd     = torch.load(ckpt_path, weights_only=True)
    arch   = detect_arch(sd)
    if arch == "unknown":
        log.warning(f"  Unknown arch: {ckpt_path.name}")
        return None

    method    = info["method"]
    test_fold = info["test_fold"]
    edge_in   = _get_edge_in(sd, arch)
    hidden    = _get_hidden(sd, arch)
    node_in   = _get_node_in(sd, arch, hidden)
    within    = "B3_within" in method

    if edge_in == 2 and "hybrid" in method.lower():
        return process_e64_hybrid(ckpt_path, sd, info, dev, device,
                                  node_in, edge_in, hidden)

    graph = load_test_graph(test_fold, edge_in, method, dev, within)

    try:
        if arch == "sage":
            model = EdgeAwareSAGE(node_in=node_in, edge_in=edge_in,
                                  hidden=hidden, num_classes=2)
            model.load_state_dict(sd)
            scores = _infer_sage(model, graph, device)

        elif arch == "gib":
            bn_dim = sd["to_dist.weight"].shape[0] // 2
            model  = GIB_EGraphSAGE(node_in=node_in, edge_in=edge_in,
                                     hidden=hidden, bottleneck_dim=bn_dim)
            model.load_state_dict(sd)
            scores = _infer_sage(model, graph, device)

        elif arch == "dann":
            dom_w = sd.get("domain_head.2.weight")
            nd    = dom_w.shape[0] if dom_w is not None else 3
            try:
                from run_phase4 import DANN_EGS
                model = DANN_EGS(node_in=node_in, edge_in=edge_in,
                                 hidden=hidden, num_domains=nd)
                model.load_state_dict(sd)
            except RuntimeError:
                from run_phase5 import CDAN_EGS
                model = CDAN_EGS(node_in=node_in, edge_in=edge_in,
                                 hidden=hidden, num_domains=nd)
                model.load_state_dict(sd)
            scores = _infer_dann(model, graph, device)

        elif arch == "moe":
            K = max(int(k.split(".")[1]) for k in sd if k.startswith("experts."))
            model = MoE_IDS(node_in=node_in, edge_in=edge_in, hidden=hidden, K=K)
            model.load_state_dict(sd)
            scores = _infer_sage(model, graph, device)

        else:
            return None

        labels         = graph.edge_label.numpy().copy()
        attack_classes = list(graph.edge_label_type)
    finally:
        del sd
        try:
            del model
        except NameError:
            pass
        del graph

    return scores, labels, attack_classes


def process_e64_hybrid(ckpt_path: Path, sd: dict, info: dict,
                        dev: bool, device: str,
                        node_in: int, edge_in: int, hidden: int):
    method    = info["method"]
    test_fold = info["test_fold"]
    seed      = info["seed"]

    base_m = re.search(r'E6\.(\d+)', method)
    if not base_m:
        log.warning(f"  Cannot determine base for {method}")
        return None
    base_prefix = "E6.2_msa" if base_m.group(1) == "2" else "E6.1_anomal_e"

    enc_path = MODELS_DIR / f"{base_prefix}_encoder_seed{seed}_test{test_fold}.pt"
    if_path  = MODELS_DIR / f"{base_prefix}_if_seed{seed}_test{test_fold}.pt"
    if not enc_path.exists() or not if_path.exists():
        log.warning(f"  Missing base encoder/IF for {method}")
        return None

    enc_sd  = torch.load(enc_path, weights_only=True)
    enc_ein = enc_sd["edge_enc.0.weight"].shape[1]
    enc_h   = enc_sd["edge_enc.0.weight"].shape[0]
    enc_nin = enc_sd["conv1.lin_l.weight"].shape[1] - enc_h
    encoder = EdgeAwareSAGE(node_in=enc_nin, edge_in=enc_ein, hidden=enc_h)
    encoder.load_state_dict(enc_sd)
    del enc_sd
    iforest = torch.load(if_path, weights_only=False)

    graph_a     = load_graph(test_fold, tier="A", dev=dev)
    anom_scores = _anomaly_scores_batched(encoder, iforest, graph_a, device)
    del encoder, iforest

    # Build hybrid [1.0, score] features in-place on a copy
    sc  = torch.as_tensor(anom_scores, dtype=torch.float32).unsqueeze(1)
    ea  = torch.cat([torch.ones_like(sc), sc], dim=1)
    graph_h = copy.copy(graph_a)
    graph_h.edge_attr   = ea
    graph_h.edge_attr_q = ea
    del anom_scores, sc, ea

    model = EdgeAwareSAGE(node_in=node_in, edge_in=edge_in, hidden=hidden)
    model.load_state_dict(sd)
    scores = _infer_sage(model, graph_h, device)
    del model

    labels         = graph_a.edge_label.numpy().copy()
    attack_classes = list(graph_a.edge_label_type)
    del graph_a, graph_h
    return scores, labels, attack_classes


def process_anomaly_encoder(ckpt_path: Path, info: dict, dev: bool, device: str):
    test_fold = info["test_fold"]

    if_path = ckpt_path.parent / ckpt_path.name.replace("_encoder_", "_if_")
    if not if_path.exists():
        log.warning(f"  IF pickle not found: {if_path}")
        return None

    sd      = torch.load(ckpt_path, weights_only=True)
    edge_in = sd["edge_enc.0.weight"].shape[1]
    hidden  = sd["edge_enc.0.weight"].shape[0]
    node_in = sd["conv1.lin_l.weight"].shape[1] - hidden
    encoder = EdgeAwareSAGE(node_in=node_in, edge_in=edge_in, hidden=hidden)
    encoder.load_state_dict(sd)
    del sd
    iforest = torch.load(if_path, weights_only=False)

    graph   = load_graph(test_fold, tier="A", dev=dev)
    scores  = _anomaly_scores_batched(encoder, iforest, graph, device)
    del encoder, iforest

    labels         = graph.edge_label.numpy().copy()
    attack_classes = list(graph.edge_label_type)
    del graph
    return scores, labels, attack_classes


def process_gate(ckpt_path: Path, info: dict, dev: bool, device: str):
    test_fold   = info["test_fold"]
    seed        = info["seed"]
    train_dsets = FOLD_TRAIN[test_fold]
    n_src       = len(train_dsets)

    sd_gate     = torch.load(ckpt_path, weights_only=True)
    in_feat     = sd_gate["net.0.weight"].shape[1]
    hidden_g    = sd_gate["net.0.weight"].shape[0]
    from run_phase5 import GateNetwork
    gate = GateNetwork(in_features=in_feat, hidden=hidden_g, num_sources=n_src)
    gate.load_state_dict(sd_gate)
    gate.eval().to(device)
    del sd_gate

    from run_phase4 import DANN_EGS
    specialists = []
    for src_d in range(n_src):
        spec_name = ckpt_path.name.replace("_gate_", f"_spec{src_d}_")
        spec_path = ckpt_path.parent / spec_name
        if not spec_path.exists():
            log.warning(f"  Specialist {src_d} missing: {spec_path}")
            return None
        sd_s  = torch.load(spec_path, weights_only=True)
        ein   = sd_s["encoder.edge_enc.0.weight"].shape[1]
        h     = sd_s["encoder.edge_enc.0.weight"].shape[0]
        nin   = sd_s["encoder.conv1.lin_l.weight"].shape[1] - h
        dom_w = sd_s.get("domain_head.2.weight")
        nd    = dom_w.shape[0] if dom_w is not None else n_src
        spec  = DANN_EGS(node_in=nin, edge_in=ein, hidden=h, num_domains=nd)
        spec.load_state_dict(sd_s)
        del sd_s
        specialists.append(spec)

    g_struct  = _struct_only(load_graph(test_fold, tier="B", dev=dev))
    g_tier_a  = load_graph(test_fold, tier="A", dev=dev)
    ea_q_a    = quantile_encode(g_tier_a.edge_attr)
    E         = g_struct.edge_label.shape[0]

    # Gate weights in batches
    ea_a_dev = ea_q_a.to(device)
    gate_parts = []
    with torch.no_grad():
        for s in range(0, E, 50_000):
            gate_parts.append(gate(ea_a_dev[s:s+50_000]).cpu())
    w = torch.cat(gate_parts, dim=0)            # [E, n_src]
    del gate_parts, ea_a_dev, ea_q_a, g_tier_a

    # Specialist logits — one specialist at a time to avoid holding all simultaneously
    x_t  = g_struct.x.to(device)
    ei_t = g_struct.edge_index.to(device)
    ea_t = g_struct.edge_attr_q.to(device)

    # Build weighted sum incrementally: combined[e,c] += w[e,s] * logit[e,s,c]
    combined = torch.zeros(E, 2)
    for src_d, spec in enumerate(specialists):
        spec.eval().to(device)
        parts = []
        with torch.no_grad():
            for s in range(0, E, 50_000):
                parts.append(spec.predict(x_t, ei_t[:, s:s+50_000],
                                          ea_t[s:s+50_000]).cpu())
        logits = torch.cat(parts, dim=0)            # [E, 2]
        combined += w[:, src_d:src_d+1] * logits
        del parts, logits
        spec.cpu()                                   # move back to CPU to free device memory

    del x_t, ei_t, ea_t, w, gate, specialists

    scores         = F.softmax(combined, dim=-1)[:, 1].numpy().astype(np.float32)
    labels         = g_struct.edge_label.numpy().copy()
    attack_classes = list(g_struct.edge_label_type)
    del combined, g_struct
    return scores, labels, attack_classes


# ── atomic save ───────────────────────────────────────────────────────────────

def _atomic_save(obj: dict, out_path: Path):
    """Write to a temp file then rename so partial writes never look complete."""
    tmp_path = out_path.with_suffix(".tmp")
    torch.save(obj, tmp_path)
    tmp_path.rename(out_path)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev",       action="store_true",
                        help="Use dev-subsampled graphs (~2M edges each)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run even if output file already exists")
    parser.add_argument("--method",    default=None,
                        help="Only process checkpoints whose filename contains STRING")
    args = parser.parse_args()

    setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(MODELS_DIR.glob("*.pt"))
    log.info(f"Found {len(ckpts)} checkpoints in {MODELS_DIR}")

    done = skip = fail = 0
    for ckpt_path in ckpts:
        stem     = ckpt_path.stem
        out_path = INFERENCE_DIR / f"{stem}.pt"

        # ── filter by method string ───────────────────────────────────────────
        if args.method and args.method not in stem:
            continue

        # ── skip if already done (check before parsing to be fast) ───────────
        if out_path.exists() and not args.overwrite:
            log.info(f"  Skip (exists): {stem}")
            done += 1
            continue

        # ── parse filename ────────────────────────────────────────────────────
        info = parse_ckpt(stem)
        if info is None:
            log.warning(f"  Unparseable filename: {stem}")
            skip += 1
            continue
        if info["role"] == "skip":
            log.debug(f"  Component (skip): {stem}")
            skip += 1
            continue

        log.info(f"Processing {stem} …")
        result = None
        try:
            role = info["role"]
            if role == "anomaly_encoder":
                result = process_anomaly_encoder(ckpt_path, info, args.dev, device)
            elif role == "gate_trigger":
                result = process_gate(ckpt_path, info, args.dev, device)
            else:
                result = process_main(ckpt_path, info, args.dev, device)

            if result is None:
                log.warning(f"  Skipped (no result): {stem}")
                skip += 1
                continue

            scores, labels, attack_classes = result
            _atomic_save({
                "scores":         scores,
                "labels":         labels,
                "attack_classes": attack_classes,
                "method":         info["method"],
                "seed":           info["seed"],
                "test_fold":      info["test_fold"],
            }, out_path)
            log.info(f"  Saved {out_path.name}  "
                     f"E={len(scores):,}  attack_rate={labels.mean():.3f}  "
                     f"scores=[{scores.min():.3f},{scores.max():.3f}]")
            done += 1

        except MemoryError:
            log.error(f"  OOM on {stem} — skipping. Try --dev for smaller graphs.")
            fail += 1
        except Exception as exc:
            log.error(f"  FAILED {stem}: {exc}", exc_info=True)
            fail += 1
        finally:
            # Always clean up, even on exception
            del result
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    log.info(f"\nDone={done}  Skip={skip}  Fail={fail}")


if __name__ == "__main__":
    main()
