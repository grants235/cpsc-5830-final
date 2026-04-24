"""
GraphMETRO-inspired Mixture of Experts for cross-dataset NIDS.
K=3 shift axes: density, feature distribution, label balance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from src.models.egraphsage import EdgeAwareSAGE


class GatingNetwork(nn.Module):
    """2-layer GCN + global mean pool → softmax over K+1 experts."""

    def __init__(self, node_in: int, edge_in: int, hidden: int, num_experts: int):
        super().__init__()
        self.edge_proj = nn.Linear(edge_in, node_in)
        self.conv1 = GCNConv(node_in, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head  = nn.Linear(hidden, num_experts)

    def forward(self, x, edge_index, edge_attr, batch):
        # Broadcast mean edge features into nodes as auxiliary signal
        from torch_geometric.utils import scatter
        e_proj = self.edge_proj(edge_attr)
        row, col = edge_index
        msg = scatter(e_proj, col, dim=0, dim_size=x.size(0), reduce="mean")
        h = (x + msg).relu()
        h = self.conv1(h, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        g = global_mean_pool(h, batch)       # [B, hidden]
        return F.softmax(self.head(g), dim=-1)  # [B, K+1]


class MoE_IDS(nn.Module):
    """
    Full MoE: reference expert ξ_0 + K shift experts.
    Training loss = CE(classifier(h_mix), y) + λ * L_align.
    """

    def __init__(self, node_in: int, edge_in: int, hidden: int = 128,
                 num_classes: int = 2, K: int = 3, lam: float = 0.1,
                 dropout: float = 0.2):
        super().__init__()
        self.K   = K
        self.lam = lam

        # Reference expert (K+1 experts total including reference)
        self.experts = nn.ModuleList([
            EdgeAwareSAGE(node_in, edge_in, hidden, num_classes, dropout)
            for _ in range(K + 1)
        ])
        self.gate = GatingNetwork(node_in, edge_in, hidden, K + 1)
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, num_classes),
        )

    def _embed(self, expert_idx, x, edge_index, edge_attr):
        return self.experts[expert_idx].embed(x, edge_index, edge_attr)

    def forward(self, x, edge_index, edge_attr, batch=None,
                augmented_attrs=None, return_loss_components=False):
        """
        Args:
            augmented_attrs: list of K tensors (one per shift expert),
                             or None at test time.
            batch:           node-to-graph assignment (can be None for single graph).
        Returns:
            logits [E, C], or (logits, L_task, L_align) if return_loss_components.
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Reference embedding on unaugmented data
        h0 = self._embed(0, x, edge_index, edge_attr)  # [E, 3H]

        if augmented_attrs is not None:
            # Expert embeddings on augmented data
            expert_embeds = [
                self._embed(k + 1, x, edge_index, augmented_attrs[k])
                for k in range(self.K)
            ]
            L_align = sum(
                ((he - h0.detach()) ** 2).mean()
                for he in expert_embeds
            ) / self.K
        else:
            expert_embeds = [self._embed(k + 1, x, edge_index, edge_attr)
                             for k in range(self.K)]
            L_align = torch.tensor(0.0, device=x.device)

        # Gate weights: need per-edge assignment; use per-node batch → majority
        gate_w = self.gate(x, edge_index, edge_attr, batch)  # [1 or B, K+1]
        # Broadcast gate to edge level via source node
        row = edge_index[0]
        # For single-graph (batch all zeros): gate_w is [1, K+1]
        if gate_w.shape[0] == 1:
            gw_edge = gate_w.expand(h0.shape[0], -1)  # [E, K+1]
        else:
            node_batch = batch                          # [N]
            edge_batch = node_batch[row]                # [E]
            gw_edge = gate_w[edge_batch]                # [E, K+1]

        all_embeds = torch.stack([h0] + expert_embeds, dim=1)  # [E, K+1, 3H]
        h_mix = (gw_edge.unsqueeze(-1) * all_embeds).sum(dim=1)  # [E, 3H]
        logits = self.classifier(h_mix)

        if return_loss_components:
            return logits, L_align
        return logits

    def get_gate_weights(self, x, edge_index, edge_attr, batch=None):
        """Return gate soft-weights for interpretability analysis."""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        with torch.no_grad():
            return self.gate(x, edge_index, edge_attr, batch)


# ---------------------------------------------------------------------------
# Augmentation functions for the K=3 shift experts
# ---------------------------------------------------------------------------

def augment_density(edge_index, edge_attr, edge_label, drop_frac: float = None):
    """Randomly drop 30–70% of edges (density shift ξ₁)."""
    if drop_frac is None:
        drop_frac = torch.empty(1).uniform_(0.3, 0.7).item()
    E = edge_attr.shape[0]
    keep_mask = torch.rand(E, device=edge_attr.device) >= drop_frac
    return edge_index[:, keep_mask], edge_attr[keep_mask], edge_label[keep_mask]


def augment_feature_noise(edge_attr, sigma: float = 0.1):
    """Add Gaussian noise to quantile features, then clip to [0,1]."""
    noisy = edge_attr + torch.randn_like(edge_attr) * sigma
    return noisy.clamp(0.0, 1.0)


def augment_label_balance(edge_attr, edge_label, target_ratio: float = 0.5):
    """Oversample minority class to reach target ratio (label balance shift ξ₃)."""
    attack_idx  = (edge_label == 1).nonzero(as_tuple=True)[0]
    benign_idx  = (edge_label == 0).nonzero(as_tuple=True)[0]
    n_attack = attack_idx.shape[0]
    n_benign = benign_idx.shape[0]
    if n_attack == 0 or n_benign == 0:
        return edge_attr, edge_label
    # oversample minority
    if n_attack < n_benign:
        needed = int(n_benign * target_ratio / (1 - target_ratio)) - n_attack
        if needed > 0:
            extra = attack_idx[torch.randint(n_attack, (needed,), device=edge_attr.device)]
            all_idx = torch.cat([torch.arange(edge_attr.shape[0], device=edge_attr.device), extra])
        else:
            all_idx = torch.arange(edge_attr.shape[0], device=edge_attr.device)
    else:
        needed = int(n_attack * target_ratio / (1 - target_ratio)) - n_benign
        if needed > 0:
            extra = benign_idx[torch.randint(n_benign, (needed,), device=edge_attr.device)]
            all_idx = torch.cat([torch.arange(edge_attr.shape[0], device=edge_attr.device), extra])
        else:
            all_idx = torch.arange(edge_attr.shape[0], device=edge_attr.device)
    perm = all_idx[torch.randperm(all_idx.shape[0], device=edge_attr.device)]
    return edge_attr[perm], edge_label[perm]
