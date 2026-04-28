"""
Temporal subgraph extraction for E8.1/E8.2.

For a query edge (u, v, t) and a time delta Δ, the context subgraph is:
  - All edges e' with t' ∈ [t-Δ, t]  (the query edge itself included)
  - That touch the 2-hop temporal neighbourhood of u and v.

The edges array must be sorted in ascending timestamp order (guaranteed by
graph_builder.build_graph and combine_graphs).
"""

import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from torch_geometric.data import Data

# Pre-cap multiplier: subsample window to this many edges before BFS.
# Bounds BFS cost on dense datasets (e.g. CIC2018 with 60K+ edges per 60s window).
_BFS_PRE_CAP = 4


def extract_temporal_subgraph(
    src_np: np.ndarray,
    dst_np: np.ndarray,
    time_np: np.ndarray,
    u: int,
    v: int,
    t: int,
    delta_us: int,
    max_edges: int = 1024,
    n_hops: int = 2,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """
    Return global edge indices for the temporal context subgraph of query (u, v, t).

    Parameters
    ----------
    src_np, dst_np, time_np : sorted numpy int arrays (shape [E])
    u, v   : query source / destination node ids (global)
    t      : query timestamp (µs)
    delta_us: window size in µs  (e.g. 60_000_000 for 60 s)
    max_edges: cap on returned edges; excess sampled uniformly
    n_hops : BFS depth for neighbourhood filtering (default 2)
    rng    : optional RNG for reproducible capping

    Returns
    -------
    np.ndarray of global edge indices (int64), possibly empty.
    """
    t_lo = t - delta_us
    lo   = int(np.searchsorted(time_np, t_lo, side="left"))
    hi   = int(np.searchsorted(time_np, t,    side="right"))

    if lo >= hi:
        return np.empty(0, dtype=np.int64)

    window_size = hi - lo
    pre_cap     = max_edges * _BFS_PRE_CAP

    # Pre-subsample before BFS to bound cost on dense windows (e.g. CIC2018
    # has ~60K edges per 60s window; BFS on 60K is very slow).
    if window_size > pre_cap:
        if rng is None:
            rng = np.random.RandomState()
        pre_local = rng.choice(window_size, pre_cap, replace=False)
        pre_local.sort()
        w_src       = src_np[lo + pre_local]
        w_dst       = dst_np[lo + pre_local]
        global_base = (lo + pre_local).astype(np.int64)
    else:
        w_src       = src_np[lo:hi]
        w_dst       = dst_np[lo:hi]
        global_base = np.arange(lo, hi, dtype=np.int64)

    # 2-hop BFS from (u, v) within the (pre-capped) window
    node_set = np.array([u, v], dtype=np.int64)
    for _ in range(n_hops):
        inc = np.isin(w_src, node_set) | np.isin(w_dst, node_set)
        if not inc.any():
            break
        node_set = np.unique(np.concatenate([node_set, w_src[inc], w_dst[inc]]))

    # Keep edges where either endpoint is in node_set
    final_inc  = np.isin(w_src, node_set) | np.isin(w_dst, node_set)
    local_idx  = np.where(final_inc)[0]
    global_idx = global_base[local_idx]

    # Cap at max_edges
    if len(global_idx) > max_edges:
        if rng is None:
            rng = np.random.RandomState()
        chosen     = rng.choice(len(global_idx), max_edges, replace=False)
        global_idx = global_idx[chosen]

    return global_idx.astype(np.int64)


def build_subgraph_data(
    src_np: np.ndarray,
    dst_np: np.ndarray,
    global_idx: np.ndarray,
    query_u: int,
    query_v: int,
    node_feat_dim: int = 8,
) -> Data:
    """
    Build a PyG Data object for a single temporal context subgraph.

    The query nodes (query_u, query_v) are always included in the node set
    even if they appear in no context edge.  Their LOCAL indices within this
    subgraph are stored as data.query_u / data.query_v for retrieval after
    PyG Batch stacking.

    Edge features are constant 1.0 (structure-only, matching E1.E).
    """
    if len(global_idx) > 0:
        sub_src = src_np[global_idx]
        sub_dst = dst_np[global_idx]
        all_nodes = np.unique(np.concatenate([sub_src, sub_dst,
                                               [query_u, query_v]]))
    else:
        all_nodes = np.array([query_u, query_v], dtype=np.int64)
        sub_src   = np.empty(0, dtype=np.int64)
        sub_dst   = np.empty(0, dtype=np.int64)

    # Map global → local node indices via sorted-array binary search
    local_src = np.searchsorted(all_nodes, sub_src).astype(np.int64)
    local_dst = np.searchsorted(all_nodes, sub_dst).astype(np.int64)
    qu_local  = int(np.searchsorted(all_nodes, query_u))
    qv_local  = int(np.searchsorted(all_nodes, query_v))

    N  = len(all_nodes)
    E  = len(global_idx)
    ei = torch.as_tensor(np.stack([local_src, local_dst]) if E > 0
                         else np.zeros((2, 0), dtype=np.int64), dtype=torch.long)
    ea = torch.ones(E, 1, dtype=torch.float32)
    x  = torch.ones(N, node_feat_dim, dtype=torch.float32)

    data         = Data(x=x, edge_index=ei, edge_attr=ea)
    data.query_u = qu_local
    data.query_v = qv_local
    return data


def batch_build_subgraphs(
    src_np: np.ndarray,
    dst_np: np.ndarray,
    time_np: np.ndarray,
    query_u_arr: np.ndarray,
    query_v_arr: np.ndarray,
    query_t_arr: np.ndarray,
    delta_us: int,
    max_edges: int = 1024,
    node_feat_dim: int = 8,
    seed: int = 0,
    n_jobs: int = 4,
) -> list:
    """
    Build a list of Data objects for a batch of query edges.
    Returned list has the same length as query_u_arr.

    n_jobs: worker threads for parallel extraction (NumPy releases the GIL).
    """
    n    = len(query_u_arr)
    rng  = np.random.RandomState(seed)
    # Give each worker its own seed so RNG choices don't collide
    per_seeds = rng.randint(0, 2**31, size=n)

    def _one(i: int) -> Data:
        rng_i = np.random.RandomState(int(per_seeds[i]))
        gidx  = extract_temporal_subgraph(
            src_np, dst_np, time_np,
            int(query_u_arr[i]), int(query_v_arr[i]), int(query_t_arr[i]),
            delta_us, max_edges, rng=rng_i,
        )
        return build_subgraph_data(
            src_np, dst_np, gidx,
            int(query_u_arr[i]), int(query_v_arr[i]),
            node_feat_dim=node_feat_dim,
        )

    if n_jobs > 1 and n > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            data_list = list(ex.map(_one, range(n)))
    else:
        data_list = [_one(i) for i in range(n)]

    return data_list
