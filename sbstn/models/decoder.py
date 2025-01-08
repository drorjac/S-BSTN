"""LSTM decoder + linear head.

Paper: "LSTM Decoder Prediction", Fig. 2 (right block).

    h(d)_t' = LSTMCell( [y_{t'-1} ; z~_t'] , (h(d)_{t'-1}, c(d)_{t'-1}) )
    y_t'    = W_y [ z~_t' ; h(d)_t' ] + b_y

The head is deliberately LINEAR, with no activation: the paper's stated
rationale is that the earlier layers carry the nonlinearity and the final layer
maps to unbounded continuous outputs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DecoderLSTM(nn.Module):
    """One decoding step at a time, so the caller can interleave attention.

    The context `z` for step t' depends on the decoder state from t'-1, so the
    loop cannot be batched over the horizon - the caller owns the loop and
    calls `step` once per future step.
    """

    def __init__(self, n_sensors: int, hidden: int, *, residual: bool = False):
        super().__init__()
        self.N = n_sensors
        self.M = hidden
        # DEVIATION FROM THE PAPER, off by default.
        #
        # The paper's head maps [z ; h] straight to an absolute dB level, so
        # the network must reconstruct each link's baseline (30-60 dB) from
        # context every step. Persistence gets that for free, which is why it
        # is near-unbeatable at one minute. A residual head predicts the CHANGE
        # from the last value instead, making persistence the trivial solution
        # the model starts from. Worth measuring; not the specified design.
        self.residual = residual
        self.cell = nn.LSTMCell(n_sensors + hidden, hidden)
        # W_y in R^{N x 2M}: consumes [context ; decoder state].
        self.head = nn.Linear(2 * hidden, n_sensors)

    def step(self, y_prev: torch.Tensor, z: torch.Tensor,
             h: torch.Tensor, c: torch.Tensor):
        """y_prev: (B, N), z: (B, M) -> y_hat: (B, N), and the new state."""
        h, c = self.cell(torch.cat([y_prev, z], dim=-1), (h, c))
        y_hat = self.head(torch.cat([z, h], dim=-1))
        if self.residual:
            y_hat = y_prev + y_hat
        return y_hat, h, c


class DecoderInit(nn.Module):
    """Project the final encoder states into the decoder's initial state.

    Starting the decoder from zeros throws away everything the encoder just
    computed, so the final states are projected instead.
    """

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.init_h = nn.Linear(in_dim, hidden)
        self.init_c = nn.Linear(in_dim, hidden)

    def forward(self, h0: torch.Tensor, c0: torch.Tensor):
        return torch.tanh(self.init_h(h0)), torch.tanh(self.init_c(c0))
