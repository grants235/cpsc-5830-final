"""
Temporal GNN models for E8.1/E8.2.

TemporalEdgeSAGE (E8.1):
  Runs standard SAGE on the per-query temporal subgraph, extracts
  (h_u, h_v, enc(q_ea)) for the query edge, classifies with MLP.

TemporalIDGNN (E8.2):
  Same but runs message passing TWICE per batch — once with the query
  source node tagged as anchor, once with the destination tagged — then
  combines both perspectives before classification.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import scatter


class TemporalEdgeSAGE(nn.Module):
    def __init__(self, node_in: int = 8, edge_in: int = 1,
                 hidden: int = 128, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden = hidden
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

    def _node_embeds(self, x: torch.Tensor, edge_index: torch.Tensor,
                     edge_attr: torch.Tensor) -> torch.Tensor:
        """SAGE on subgraph; returns node embeddings [N, H]."""
        e   = self.edge_enc(edge_attr)
        row, col = edge_index
        msg = scatter(e, col, dim=0, dim_size=x.size(0), reduce="mean")
        h   = torch.cat([x, msg], dim=-1)
        h   = self.norm1(self.conv1(h, edge_index).relu())
        h   = self.norm2(self.conv2(h, edge_index).relu())
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                u_globals: torch.Tensor, v_globals: torch.Tensor,
                q_ea: torch.Tensor) -> torch.Tensor:
        """
        x, edge_index, edge_attr : temporal context subgraph (batched)
        u_globals, v_globals     : [B] global node indices of query src/dst
        q_ea                     : [B, edge_in] query edge features
        Returns logits [B, num_classes].
        """
        h   = self._node_embeds(x, edge_index, edge_attr)
        e_q = self.edge_enc(q_ea)
        z   = torch.cat([h[u_globals], h[v_globals], e_q], dim=-1)
        return self.edge_head(z)

    def embed(self, x, edge_index, edge_attr,
              u_globals, v_globals, q_ea) -> torch.Tensor:
        """Return pre-head embeddings [B, 3H]."""
        h   = self._node_embeds(x, edge_index, edge_attr)
        e_q = self.edge_enc(q_ea)
        return torch.cat([h[u_globals], h[v_globals], e_q], dim=-1)


class TemporalIDGNN(nn.Module):
    """
    ID-GNN variant: augments node features with a 1-bit identity marker,
    runs SAGE twice (u-anchor pass + v-anchor pass), combines for prediction.
    """

    def __init__(self, node_in: int = 8, edge_in: int = 1,
                 hidden: int = 128, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden   = hidden
        self.node_in  = node_in
        aug_in = node_in + 1          # +1 for the identity bit
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_in, hidden), nn.ReLU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
        )
        self.conv1 = SAGEConv(aug_in + hidden, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, hidden, aggr="mean")
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.edge_head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, num_classes),
        )

    def _node_embeds_with_anchor(self, x: torch.Tensor,
                                  edge_index: torch.Tensor,
                                  edge_attr: torch.Tensor,
                                  anchor_globals: torch.Tensor) -> torch.Tensor:
        """
        Append identity bit (1 at anchor nodes, 0 elsewhere) to x,
        then run SAGE. Returns node embeddings [N, H].
        anchor_globals: [B] global indices of the anchor nodes.
        """
        N   = x.size(0)
        bit = torch.zeros(N, 1, device=x.device, dtype=x.dtype)
        bit[anchor_globals] = 1.0
        x_aug = torch.cat([x, bit], dim=-1)     # [N, node_in+1]

        e   = self.edge_enc(edge_attr)
        row, col = edge_index
        msg = scatter(e, col, dim=0, dim_size=N, reduce="mean")
        h   = torch.cat([x_aug, msg], dim=-1)
        h   = self.norm1(self.conv1(h, edge_index).relu())
        h   = self.norm2(self.conv2(h, edge_index).relu())
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                u_globals: torch.Tensor, v_globals: torch.Tensor,
                q_ea: torch.Tensor) -> torch.Tensor:
        e_q = self.edge_enc(q_ea)
        h1  = self._node_embeds_with_anchor(x, edge_index, edge_attr, u_globals)
        h_u = h1[u_globals]
        h2  = self._node_embeds_with_anchor(x, edge_index, edge_attr, v_globals)
        h_v = h2[v_globals]
        z   = torch.cat([h_u, h_v, e_q], dim=-1)
        return self.edge_head(z)

    def embed(self, x, edge_index, edge_attr,
              u_globals, v_globals, q_ea) -> torch.Tensor:
        e_q = self.edge_enc(q_ea)
        h1  = self._node_embeds_with_anchor(x, edge_index, edge_attr, u_globals)
        h_u = h1[u_globals]
        h2  = self._node_embeds_with_anchor(x, edge_index, edge_attr, v_globals)
        h_v = h2[v_globals]
        return torch.cat([h_u, h_v, e_q], dim=-1)
