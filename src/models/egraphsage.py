"""
E-GraphSAGE for edge classification (Lo et al. 2022).
PyG implementation: edge features pre-aggregated into node state,
then 2 SAGE layers, then edge head on [h_src, h_dst, edge_enc].
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import scatter


class EdgeAwareSAGE(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden: int = 128,
                 num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_in, hidden), nn.ReLU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
        )
        self.conv1 = SAGEConv(node_in + hidden, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, hidden, aggr="mean")
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.edge_head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, num_classes),
        )

    def forward(self, x, edge_index, edge_attr):
        e = self.edge_enc(edge_attr)                              # [E, H]
        row, col = edge_index
        msg = scatter(e, col, dim=0, dim_size=x.size(0), reduce="mean")
        h = torch.cat([x, msg], dim=-1)                           # [N, node_in+H]
        h = self.norm1(self.conv1(h, edge_index).relu())          # [N, H]
        h = self.norm2(self.conv2(h, edge_index).relu())          # [N, H]
        z = torch.cat([h[row], h[col], e], dim=-1)                # [E, 3H]
        return self.edge_head(z)                                   # [E, C]

    def embed(self, x, edge_index, edge_attr):
        """Return pre-classifier edge embeddings [E, 3H]."""
        e = self.edge_enc(edge_attr)
        row, col = edge_index
        msg = scatter(e, col, dim=0, dim_size=x.size(0), reduce="mean")
        h = torch.cat([x, msg], dim=-1)
        h = self.norm1(self.conv1(h, edge_index).relu())
        h = self.norm2(self.conv2(h, edge_index).relu())
        return torch.cat([h[row], h[col], e], dim=-1)
