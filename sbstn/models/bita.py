"""Bi-Temporal Attention (Bi-TA).

Paper: bi-temporal attention equations, Fig. 4
(assets/paper/fig4_bita_temporal.png).

U_d is implemented as Q_d x M so that U_d h_enc lands in R^{Q_d}; see below.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """One direction. Scores every encoder step against the decoder state."""

    def __init__(self, hidden: int, q_d: int):
        super().__init__()
        self.W_d = nn.Linear(2 * hidden, q_d, bias=False)   # (Q_d, 2M)
        # U_d @ h_enc must land in R^{Q_d} to be added to W_d @ [c;h], so
        # U_d is Q_d x M. This coincides with an M x M matrix when Q_d == M.
        self.U_d = nn.Linear(hidden, q_d, bias=False)       # (Q_d, M)
        self.b_d = nn.Parameter(torch.zeros(q_d))
        self.v_d = nn.Linear(q_d, 1, bias=False)            # (Q_d,)

    def forward(self, h_enc, h_dec, c_dec):
        """h_enc: (B, T, M), h_dec/c_dec: (B, M) -> z: (B, M), pi: (B, T)."""
        s = self.W_d(torch.cat([c_dec, h_dec], dim=-1))     # (B, Q_d)
        zeta = self.v_d(
            torch.tanh(s[:, None, :] + self.U_d(h_enc) + self.b_d)
        ).squeeze(-1)                                        # (B, T)
        pi = torch.softmax(zeta, dim=-1)                     # over T
        z = (pi[..., None] * h_enc).sum(dim=1)               # (B, M)
        return z, pi


class BiTA(nn.Module):
    """Per-direction temporal attention, fused once by a single FC layer.

    That single late fusion is the point: merging the directions earlier is
    exactly ablation variant V2.
    """

    def __init__(self, hidden: int, q_d: int, *, bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
        self.fwd = TemporalAttention(hidden, q_d)
        self.bwd = TemporalAttention(hidden, q_d) if bidirectional else None
        in_dim = 2 * hidden if bidirectional else hidden
        self.fuse = nn.Linear(in_dim, hidden)

    def forward(self, h_f, h_b, h_dec, c_dec):
        z_f, pi_f = self.fwd(h_f, h_dec, c_dec)
        if self.bwd is None:
            return torch.tanh(self.fuse(z_f)), pi_f, None
        z_b, pi_b = self.bwd(h_b, h_dec, c_dec)
        z = torch.tanh(self.fuse(torch.cat([z_f, z_b], dim=-1)))
        return z, pi_f, pi_b


class LastStateContext(nn.Module):
    """Ablation V2: no temporal attention.

    Context is the concatenated final encoder states, constant across t'.
    """

    def __init__(self, hidden: int, *, bidirectional: bool = True):
        super().__init__()
        in_dim = 2 * hidden if bidirectional else hidden
        self.fuse = nn.Linear(in_dim, hidden)
        self.bidirectional = bidirectional

    def forward(self, h_f, h_b, h_dec, c_dec):
        last_f = h_f[:, -1]
        if h_b is None or not self.bidirectional:
            return torch.tanh(self.fuse(last_f)), None, None
        return torch.tanh(self.fuse(torch.cat([last_f, h_b[:, 0]], dim=-1))), None, None
