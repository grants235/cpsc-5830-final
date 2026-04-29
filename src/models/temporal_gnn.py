"""
Temporal GNN models for E8.1/E8.2/E12.

TemporalEdgeSAGE (E8.1):
  Runs standard SAGE on the per-query temporal subgraph, extracts
  (h_u, h_v, enc(q_ea)) for the query edge, classifies with MLP.

TemporalIDGNN (E8.2):
  Same but runs message passing TWICE per batch — once with the query
  source node tagged as anchor, once with the destination tagged — then
  combines both perspectives before classification.

TS_GIB (E12.1/E12.2/E12.3/E12.4):
  TemporalEdgeSAGE backbone with optional variational bottleneck (GIB-style)
  and optional domain-classification auxiliary head (λ=0, multi-task).
  Supports separate query-edge encoder for when query features differ from
  context features (e.g. anomaly score appended in E12.2/E12.4).

  forward() returns (logits, kl_scalar).
  embed() returns mu (bottleneck) or 3H raw embedding (no bottleneck).
  forward_with_domain() returns (attack_logits, domain_logits, kl_scalar).
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


class TS_GIB(nn.Module):
    """
    Temporal-Subgraph + optional Variational Information Bottleneck + optional domain head.

    Constructor flags select the E12 variant:
      E12.1 — use_bottleneck=True,  num_domains=0, q_edge_in=1
      E12.2 — use_bottleneck=True,  num_domains=0, q_edge_in=2  (anomaly scalar appended)
      E12.3 — use_bottleneck=False, num_domains=3, q_edge_in=1  (TS-SAGE + λ=0 aux head)
      E12.4 — use_bottleneck=True,  num_domains=3, q_edge_in=2  (full combination)

    forward() → (logits [B, C], kl scalar)
    forward_with_domain() → (attack_logits, domain_logits, kl)  [needs domain head]
    embed() → z [B, D]  where D=hidden if bottleneck else 3*hidden
    """

    def __init__(self, node_in: int = 8, ctx_edge_in: int = 1, q_edge_in: int = 1,
                 hidden: int = 128, num_classes: int = 2, dropout: float = 0.2,
                 use_bottleneck: bool = True, num_domains: int = 0):
        super().__init__()
        self.hidden          = hidden
        self.use_bottleneck  = use_bottleneck
        self.num_domains     = num_domains

        # Context-subgraph edge encoder (always structure-only: ctx_edge_in=1)
        self.ctx_enc = nn.Sequential(
            nn.Linear(ctx_edge_in, hidden), nn.ReLU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
        )
        self.conv1 = SAGEConv(node_in + hidden, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, hidden, aggr="mean")
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        # Query-edge encoder — reuses ctx_enc when dims match, else separate
        self.q_enc = (
            None if q_edge_in == ctx_edge_in
            else nn.Sequential(
                nn.Linear(q_edge_in, hidden), nn.ReLU(),
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
            )
        )

        # Variational bottleneck: 3H → (mu, log_sigma) each H
        if use_bottleneck:
            self.to_dist = nn.Linear(3 * hidden, 2 * hidden)
            head_in = hidden
        else:
            self.to_dist = None
            head_in = 3 * hidden

        # Attack classification head
        self.edge_head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, num_classes),
        )

        # Domain auxiliary head (λ=0 multi-task; no gradient reversal)
        self.domain_head = (
            nn.Sequential(
                nn.Linear(head_in, hidden), nn.ReLU(),
                nn.Linear(hidden, num_domains),
            ) if num_domains > 0 else None
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _ctx_embeds(self, x: torch.Tensor, edge_index: torch.Tensor,
                    edge_attr: torch.Tensor) -> torch.Tensor:
        """SAGE on context subgraph → node embeddings [N, H]."""
        e        = self.ctx_enc(edge_attr)
        row, col = edge_index
        msg      = scatter(e, col, dim=0, dim_size=x.size(0), reduce="mean")
        h        = torch.cat([x, msg], dim=-1)
        h        = self.norm1(self.conv1(h, edge_index).relu())
        h        = self.norm2(self.conv2(h, edge_index).relu())
        return h

    def _q_emb(self, q_ea: torch.Tensor) -> torch.Tensor:
        enc = self.q_enc if self.q_enc is not None else self.ctx_enc
        return enc(q_ea)

    def _raw_embed(self, x, edge_index, edge_attr,
                   u_globals, v_globals, q_ea) -> torch.Tensor:
        """3H raw edge embedding before bottleneck."""
        h   = self._ctx_embeds(x, edge_index, edge_attr)
        e_q = self._q_emb(q_ea)
        return torch.cat([h[u_globals], h[v_globals], e_q], dim=-1)  # [B, 3H]

    def _bottleneck(self, z_raw: torch.Tensor):
        """Variational bottleneck: [B, 3H] → (z [B, H], kl scalar)."""
        params     = self.to_dist(z_raw)
        mu, log_s  = params.chunk(2, dim=-1)
        kl         = -0.5 * (1.0 + log_s - mu.pow(2) - log_s.exp()).sum(dim=-1).mean()
        z          = (mu + torch.exp(0.5 * log_s) * torch.randn_like(mu)
                      if self.training else mu)
        return z, kl

    # ── public forward methods ────────────────────────────────────────────────

    def forward(self, x, edge_index, edge_attr,
                u_globals, v_globals, q_ea):
        """Returns (logits [B, C], kl scalar)."""
        z_raw = self._raw_embed(x, edge_index, edge_attr, u_globals, v_globals, q_ea)
        if self.use_bottleneck:
            z, kl = self._bottleneck(z_raw)
        else:
            z, kl = z_raw, torch.tensor(0.0, device=x.device)
        return self.edge_head(z), kl

    def forward_with_domain(self, x, edge_index, edge_attr,
                            u_globals, v_globals, q_ea):
        """Returns (attack_logits, domain_logits | None, kl scalar)."""
        z_raw = self._raw_embed(x, edge_index, edge_attr, u_globals, v_globals, q_ea)
        if self.use_bottleneck:
            z, kl = self._bottleneck(z_raw)
        else:
            z, kl = z_raw, torch.tensor(0.0, device=x.device)
        atk = self.edge_head(z)
        dom = self.domain_head(z) if self.domain_head is not None else None
        return atk, dom, kl

    def embed(self, x, edge_index, edge_attr,
              u_globals, v_globals, q_ea) -> torch.Tensor:
        """Return z [B, D] for probing: mu (bottleneck) or 3H raw."""
        z_raw = self._raw_embed(x, edge_index, edge_attr, u_globals, v_globals, q_ea)
        if self.use_bottleneck:
            mu, _ = self.to_dist(z_raw).chunk(2, dim=-1)
            return mu
        return z_raw
