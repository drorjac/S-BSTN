"""Min-max scaling, fit on TRAIN ONLY."""
from __future__ import annotations
import numpy as np


class MinMax:
    def __init__(self, per_link: bool = True):
        self.per_link = per_link
        self.lo = self.hi = None

    def fit(self, x: np.ndarray):          # x: (S, N, T) or (N, L)
        ax = (0, 2) if x.ndim == 3 else (1,)
        if self.per_link:
            self.lo = x.min(axis=ax, keepdims=True)
            self.hi = x.max(axis=ax, keepdims=True)
        else:
            self.lo, self.hi = x.min(), x.max()
        return self

    def transform(self, x):
        return (x - self.lo) / np.maximum(self.hi - self.lo, 1e-9)

    def inverse(self, x):
        """Metrics go in dB. Always invert before measuring - values computed
        on the [0,1] scale are off by the scale factor and comparable to
        nothing in the paper."""
        return x * np.maximum(self.hi - self.lo, 1e-9) + self.lo
