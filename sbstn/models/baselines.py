"""Non-learned baselines."""
from __future__ import annotations
import torch.nn as nn


class Persistence(nn.Module):
    """y_hat[t'] = x[T] for all t'.

    The floor. If the model does not beat this during rain, nothing else you
    measure means anything. The paper notes dry-period errors sit below
    quantisation for every model INCLUDING this one.
    """

    def __init__(self, horizon: int):
        super().__init__()
        self.H = horizon

    def forward(self, X, Y=None, teacher_forcing_ratio: float = 0.0):
        return X[:, :, -1:].repeat(1, 1, self.H)
