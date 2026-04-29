"""
Graph Information Bottleneck variant of E-GraphSAGE (GIB-EGS).

Architecture follows Wu et al. (NeurIPS 2020) / Alemi et al. 2017 (VIB):
  - Same SAGE backbone as EdgeAwareSAGE
  - 3H per-edge embedding projected to (mu, log_sigma) for the stochastic bottleneck
  - During training: z ~ N(mu, sigma) via reparameterization
  - During eval: z = mu (deterministic)
  - forward()       → logits  (compatible with eval_egraphsage)
  - forward_train() → (logits, kl_mean)
  - embed()         → mu  (for linear probe / downstream analysis)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import scatter


class GIB_EGraphSAGE(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden: int = 128,
                 bottleneck_dim: int = 128, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden         = hidden
        self.bottleneck_dim = bottleneck_dim

        self.edge_enc = nn.Sequential(
            nn.Linear(edge_in, hidden), nn.ReLU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
        )
        self.conv1 = SAGEConv(node_in + hidden, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, hidden, aggr="mean")
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        # Project 3H edge embedding → (mu, log_sigma) of shape [E, bottleneck_dim] each
        self.to_dist = nn.Linear(3 * hidden, 2 * bottleneck_dim)

        self.edge_head = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, num_classes),
        )

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """SAGE forward; returns 3H per-edge embedding."""
        e        = self.edge_enc(edge_attr)
        row, col = edge_index
        msg      = scatter(e, col, dim=0, dim_size=x.size(0), reduce="mean")
        h        = torch.cat([x, msg], dim=-1)
        h        = self.norm1(self.conv1(h, edge_index).relu())
        h        = self.norm2(self.conv2(h, edge_index).relu())
        return torch.cat([h[row], h[col], e], dim=-1)           # [E, 3H]

    def _bottleneck(self, h_edge: torch.Tensor):
        """
        Apply variational bottleneck to per-edge embeddings.
        Returns (z, kl_mean).  kl_mean is mean KL per edge (scalar).
        """
        params    = self.to_dist(h_edge)                        # [E, 2*D]
        mu, log_s = params.chunk(2, dim=-1)                     # [E, D] each

        kl = -0.5 * (1.0 + log_s - mu.pow(2) - log_s.exp()).sum(dim=-1).mean()

        if self.training:
            z = mu + torch.exp(0.5 * log_s) * torch.randn_like(mu)
        else:
            z = mu
        return z, kl

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """Standard forward — returns logits [E, C] (compatible with eval_egraphsage)."""
        h_edge    = self._encode(x, edge_index, edge_attr)
        z, _      = self._bottleneck(h_edge)
        return self.edge_head(z)

    def forward_train(self, x: torch.Tensor, edge_index: torch.Tensor,
                      edge_attr: torch.Tensor):
        """Training forward — returns (logits [E, C], kl scalar)."""
        h_edge    = self._encode(x, edge_index, edge_attr)
        z, kl     = self._bottleneck(h_edge)
        return self.edge_head(z), kl

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor,
              edge_attr: torch.Tensor) -> torch.Tensor:
        """Return mu (deterministic bottleneck) [E, D] for probing / analysis."""
        h_edge    = self._encode(x, edge_index, edge_attr)
        params    = self.to_dist(h_edge)
        mu, _     = params.chunk(2, dim=-1)
        return mu
