"""Bidirectional spatial-attention LSTM encoder.

Paper: encoder update equations, Fig. 2 (middle block).

Tested by tests/test_model.py::test_backward_states_in_real_time_order.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .sca import SCA


class DirectionalEncoder(nn.Module):
    """One time direction: its own SCA, its own LSTM cell, no sharing.

    Deliberately NOT nn.LSTM(bidirectional=True). Each direction needs its own
    spatial attention, which that API cannot express, and the separation is the
    whole content of the paper's bi-attention claim.
    """

    def __init__(self, n_sensors, t_window, hidden, q_e, *, reverse: bool, **sca_kw):
        super().__init__()
        self.reverse = reverse
        self.M = hidden
        self.T = t_window
        self.sca = SCA(n_sensors, t_window, hidden, q_e, **sca_kw)
        self.cell = nn.LSTMCell(self.sca.out_dim, hidden)

    def forward(self, X: torch.Tensor):
        """X: (B, N, T) -> h_seq: (B, T, M), lambdas: (B, T, N, N).

        Outputs are indexed by REAL TIME in both directions. For the backward
        pass that means writing into slot t while traversing in reverse. Skip
        this and the model still trains, but every attention map you plot is
        time-reversed and the direction analysis is meaningless.
        """
        B, N, T = X.shape
        h = X.new_zeros(B, self.M)
        c = X.new_zeros(B, self.M)

        P, Q = self.sca.project(X)                      # (B, N, Q_e) each, ONCE

        h_seq = X.new_zeros(B, T, self.M)
        lams = X.new_zeros(B, T, N, N)
        empty_rates = []

        steps = range(T - 1, -1, -1) if self.reverse else range(T)
        for t in steps:
            lam, feat = self.sca(P, Q, h, c, X[:, :, t])
            h, c = self.cell(feat, (h, c))
            h_seq[:, t] = h                              # real-time index
            lams[:, t] = lam
            empty_rates.append(self.sca.last_empty_row_rate)

        self.last_empty_row_rate = sum(empty_rates) / max(len(empty_rates), 1)
        self.last_sparsity = self.sca.last_sparsity
        return h_seq, lams, (h, c)


class NoAttentionEncoder(nn.Module):
    """Ablation V1 / LSTM-ED baseline: raw x_t straight into the LSTM."""

    def __init__(self, n_sensors, t_window, hidden, *, reverse: bool):
        super().__init__()
        self.reverse = reverse
        self.M = hidden
        self.cell = nn.LSTMCell(n_sensors, hidden)

    def forward(self, X):
        B, N, T = X.shape
        h = X.new_zeros(B, self.M)
        c = X.new_zeros(B, self.M)
        h_seq = X.new_zeros(B, T, self.M)
        steps = range(T - 1, -1, -1) if self.reverse else range(T)
        for t in steps:
            h, c = self.cell(X[:, :, t], (h, c))
            h_seq[:, t] = h
        self.last_empty_row_rate = 0.0
        self.last_sparsity = 1.0
        return h_seq, None, (h, c)


class BiEncoder(nn.Module):
    """Both directions, or just forward when bidirectional=False (variant V4).

    NOTE ON LEAKAGE: both directions read only X, the observed past window.
    The backward pass is a reversed traversal of already-observed data, not a
    look at Y. There is no leakage.
    """

    def __init__(
        self,
        n_sensors,
        t_window,
        hidden,
        q_e,
        *,
        bidirectional: bool = True,
        use_spatial_attention: bool = True,
        **sca_kw,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        def make(reverse):
            if use_spatial_attention:
                return DirectionalEncoder(
                    n_sensors, t_window, hidden, q_e, reverse=reverse, **sca_kw
                )
            return NoAttentionEncoder(n_sensors, t_window, hidden, reverse=reverse)

        self.fwd = make(False)
        self.bwd = make(True) if bidirectional else None

    def forward(self, X):
        h_f, lam_f, (hf, cf) = self.fwd(X)
        if self.bwd is None:
            return h_f, None, lam_f, None, (hf, cf), (hf, cf)
        h_b, lam_b, (hb, cb) = self.bwd(X)
        return h_f, h_b, lam_f, lam_b, (hf, cf), (hb, cb)
