"""A3T-GCN style spatio-temporal graph network, written in plain PyTorch.

How it reads the data, per forecast:
  1. Encoder: for each of the last 56 days, a graph-convolutional GRU cell (T-GCN) updates
     every store's hidden state from its own sales, the calendar, and its neighbours' states
     (one hop through the normalised adjacency matrix per day, so information spreads
     across the network as the days roll forward).
  2. Attention over those 56 hidden states picks out the days that matter most (A3T-GCN).
  3. Decoder: for each of the next 28 days, an MLP combines the store's attended context,
     its final state, the known calendar of that day, a days-ahead embedding, and a
     same-weekday "seasonal naive" estimate, and outputs all product families at once.

Outputs are in each series' own scale (sales / average daily sales), always >= 0.

Reference: Bai et al., "A3T-GCN: Attention Temporal Graph Convolutional Network for
Traffic Forecasting" (2021); Zhao et al., "T-GCN" (2019).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ml.models.stgnn.data import CAL_FEATURES, HORIZON, LOOKBACK


class GraphConv(nn.Module):
    """y = (A_hat x) W + b, with A_hat the normalised adjacency (dense; the graph is small)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:  # x [B, N, D]
        return self.lin(torch.einsum("ij,bjd->bid", adj, x))


class TGCNCell(nn.Module):
    """GRU cell whose gates are graph convolutions."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.gates = GraphConv(in_dim + hidden, 2 * hidden)
        self.candidate = GraphConv(in_dim + hidden, hidden)

    def forward(self, x: torch.Tensor, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], dim=-1), adj)).chunk(2, dim=-1)
        c = torch.tanh(self.candidate(torch.cat([x, r * h], dim=-1), adj))
        return z * h + (1 - z) * c


def same_weekday_index() -> np.ndarray:
    """[HORIZON, 4] positions in the lookback window of the 4 most recent known same weekdays.

    Lookback position j holds day (origin - (LOOKBACK - 1 - j)); forecast step k is day origin + k.
    The m-th weekly lag (7m days back, m >= ceil(k / 7)) is at j = LOOKBACK - 1 + k - 7m.
    """
    idx = np.zeros((HORIZON, 4), dtype=np.int64)
    for k in range(1, HORIZON + 1):
        m0 = -(-k // 7)
        idx[k - 1] = [LOOKBACK - 1 + k - 7 * m for m in range(m0, m0 + 4)]
    assert idx.min() >= 0 and idx.max() < LOOKBACK
    return idx


class STGNN(nn.Module):
    def __init__(self, n_nodes: int, n_products: int, hidden: int = 64, node_dim: int = 8,
                 step_dim: int = 8, head_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.node_emb = nn.Embedding(n_nodes, node_dim)
        self.cell = TGCNCell(n_products + CAL_FEATURES + node_dim, hidden)
        self.attn = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.step_emb = nn.Embedding(HORIZON, step_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + CAL_FEATURES + step_dim + n_products, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, n_products),
        )
        self.register_buffer("weekday_idx", torch.from_numpy(same_weekday_index()), persistent=False)

    def forward(self, x_sales: torch.Tensor, x_cal: torch.Tensor, f_cal: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """x_sales [B, L, N, P], x_cal [B, L, C], f_cal [B, H, C], adj [N, N] -> [B, H, N, P]."""
        b, steps, n, _ = x_sales.shape
        emb = self.node_emb.weight.unsqueeze(0).expand(b, n, -1)
        h = x_sales.new_zeros(b, n, self.hidden)
        states = []
        for t in range(steps):
            cal = x_cal[:, t].unsqueeze(1).expand(b, n, -1)
            h = self.cell(torch.cat([x_sales[:, t], cal, emb], dim=-1), h, adj)
            states.append(h)
        hs = torch.stack(states, dim=1)                                  # [B, L, N, hidden]
        weights = torch.softmax(self.attn(hs), dim=1)                    # [B, L, N, 1]
        context = (weights * hs).sum(dim=1)                              # [B, N, hidden]

        naive = x_sales[:, self.weekday_idx].mean(dim=2)                 # [B, H, N, P]
        hor = f_cal.shape[1]
        feats = torch.cat([
            context.unsqueeze(1).expand(b, hor, n, -1),
            h.unsqueeze(1).expand(b, hor, n, -1),
            f_cal.unsqueeze(2).expand(b, hor, n, -1),
            self.step_emb.weight[:hor].view(1, hor, 1, -1).expand(b, hor, n, -1),
            naive,
        ], dim=-1)
        # Start from the seasonal-naive estimate and learn a correction on top of it.
        return F.softplus(self.head(feats) + naive)
