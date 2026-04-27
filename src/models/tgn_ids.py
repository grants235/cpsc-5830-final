"""
TGN-IDS: Temporal Graph Network adapted for edge (flow) classification.
Adapted from PyG examples/tgn.py — link prediction replaced with binary classifier.
"""

import types

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)


def _fix_tgn_memory_dtype(memory: TGNMemory) -> None:
    """
    PyG versions vary in whether TGNMemory.last_update is Long or Float.
    _reset_message_store always creates timestamp placeholders as Long.
    When last_update is Float (newer PyG), this causes a dtype error on the
    very first update_state call.  Patch _reset_message_store to use the same
    dtype as last_update so the two are always consistent.
    """
    lu_dtype = memory.last_update.dtype
    if lu_dtype == torch.long:
        return  # old PyG with Long last_update — no mismatch

    def _patched_reset(self):
        i   = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
        t   = self.memory.new_empty((0,), device=self.device, dtype=lu_dtype)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        self.msg_s_store = {j: (i, i, t, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, t, msg) for j in range(self.num_nodes)}

    memory._reset_message_store = types.MethodType(_patched_reset, memory)
    memory.reset_state()  # re-initialize with correct-dtype timestamp placeholders


class GraphAttentionEmbedding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 msg_dim: int, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels, out_channels // 2,
            heads=2, dropout=0.1, edge_dim=edge_dim,
        )

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class TGN_IDS(nn.Module):
    """
    TGN for intrusion detection.
    Classifies edges (flows) as benign/attack using node memory + graph attention.
    """

    def __init__(self, num_nodes: int, raw_msg_dim: int,
                 memory_dim: int = 100, time_dim: int = 100, embed_dim: int = 100):
        super().__init__()
        self.memory = TGNMemory(
            num_nodes, raw_msg_dim, memory_dim, time_dim,
            message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        _fix_tgn_memory_dtype(self.memory)
        self.gnn = GraphAttentionEmbedding(
            memory_dim, embed_dim, raw_msg_dim, self.memory.time_enc,
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim + raw_msg_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),
        )
        self.num_nodes = num_nodes

    def forward(self, src, dst, t, msg, neighbor_loader, assoc,
                t_full=None, msg_full=None):
        n_id = torch.cat([src, dst]).unique()
        n_id_sampled, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id_sampled] = torch.arange(
            n_id_sampled.size(0), device=src.device
        )
        z, last_update = self.memory(n_id_sampled)
        # e_id is a global epoch-level index from LastNeighborLoader; must index
        # into the full epoch arrays, not the current batch slice.
        _t   = t_full   if t_full   is not None else t
        _msg = msg_full if msg_full is not None else msg
        z = self.gnn(
            z, last_update, edge_index,
            _t[e_id]   if e_id.numel() > 0 else t[:0],
            _msg[e_id] if e_id.numel() > 0 else msg[:0],
        )
        edge_feat = torch.cat([z[assoc[src]], z[assoc[dst]], msg], dim=-1)
        return self.classifier(edge_feat)


class TGN_MemoryOnly(nn.Module):
    """
    E2.D ablation: no graph attention, just [memory_src, memory_dst, msg].
    """

    def __init__(self, num_nodes: int, raw_msg_dim: int,
                 memory_dim: int = 100, time_dim: int = 100):
        super().__init__()
        self.memory = TGNMemory(
            num_nodes, raw_msg_dim, memory_dim, time_dim,
            message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        _fix_tgn_memory_dtype(self.memory)
        self.classifier = nn.Sequential(
            nn.Linear(2 * memory_dim + raw_msg_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 2),
        )

    def forward(self, src, dst, t, msg, neighbor_loader=None, assoc=None):
        n_id = torch.cat([src, dst]).unique()
        z, _ = self.memory(n_id)
        assoc_local = torch.zeros(n_id.max().item() + 1, dtype=torch.long, device=src.device)
        assoc_local[n_id] = torch.arange(n_id.size(0), device=src.device)
        edge_feat = torch.cat([z[assoc_local[src]], z[assoc_local[dst]], msg], dim=-1)
        return self.classifier(edge_feat)
