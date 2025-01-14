"""Wet-dry balancing (paper, "Dataset") + sliding-window pair construction.

Pipeline:

    1. causal downsample 10 s -> 30 s
    2. rolling std per link
    3. threshold sigma_0 = q_{1-s}, FIT ON THE TRAINING PORTION ONLY
    4. wet if sd >= sigma_0
    5. any link wet over [t1, t2] -> take [t1-pre, t2+post] from ALL links
    6. slide a window -> {X (N,T), Y (N,H)} pairs
"""
from __future__ import annotations
import numpy as np


def causal_downsample(x: np.ndarray, factor: int = 3) -> np.ndarray:
    """Causal moving average then decimate. 10 s -> 30 s at factor 3.

    Causal means backward-looking only. A centred window leaks the future.
    """
    N, T = x.shape
    ker = np.ones(factor) / factor
    sm = np.stack([np.convolve(x[i], ker)[:T] for i in range(N)])
    return sm[:, factor - 1::factor]


def rolling_std(x: np.ndarray, window: int = 60) -> np.ndarray:
    """Backward-looking rolling standard deviation, per link."""
    N, T = x.shape
    sd = np.zeros_like(x, dtype=np.float64)
    for i in range(N):
        c1 = np.cumsum(np.insert(x[i].astype(np.float64), 0, 0))
        c2 = np.cumsum(np.insert(x[i].astype(np.float64) ** 2, 0, 0))
        k = np.arange(T)
        lo = np.maximum(0, k - window + 1)
        n = (k - lo + 1)
        m = (c1[k + 1] - c1[lo]) / n
        v = (c2[k + 1] - c2[lo]) / n - m ** 2
        sd[i] = np.sqrt(np.maximum(v, 0))
    return sd


def wet_dry(x: np.ndarray, window: int = 60, s: float = 0.02,
            fit_frac: float = 0.75):
    """Rolling-std wet/dry classifier (paper, "Data Aggregation").

    Uses only network data - no external calibration, no gauge.

    `fit_frac` bounds the portion of the record used to estimate the quantile
    threshold. sigma_0 is the 1-s quantile over the TRAINING portion only;
    taking it over the whole record would leak test-period statistics into a
    decision that determines which samples become training data at all.
    """
    sd = rolling_std(x, window)
    cut = max(1, int(sd.shape[1] * fit_frac))
    sigma0 = float(np.quantile(sd[:, :cut], 1.0 - s))
    return sd >= sigma0, sigma0


def balanced_intervals(wet: np.ndarray, pre: int = 20, post: int = 40):
    """Any link wet over [t1,t2] -> take [t1-pre, t2+post] from ALL links.

    The post margin matters: it captures the wet-antenna decay that
    the paper explicitly extends its windows to include.
    """
    any_wet = wet.any(0).astype(np.int8)
    T = len(any_wet)
    keep = np.zeros(T, bool)
    d = np.diff(np.concatenate([[0], any_wet, [0]]))
    for t1, t2 in zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1):
        keep[max(0, t1 - pre): min(T, t2 + post + 1)] = True
    return keep


def make_pairs(x: np.ndarray, T: int, H: int, keep: np.ndarray | None = None,
               wet_mask: np.ndarray | None = None, stride: int = 1):
    """-> X (S, N, T), Y (S, N, H), starts (S,), wet_flag (S,).

    Windows must lie fully inside a kept interval. `wet_flag[k]` is True when
    any link is genuinely raining anywhere in window k's TARGET horizon - it
    drives the wet/dry metric split, which the paper's central empirical point
    depends on (dry error sits below quantisation for every model, so a pooled
    average hides all the signal).

    Consecutive windows overlap by T+H-1 samples, so `stride` > 1 discards
    almost no information while cutting the dataset - and the training cost -
    proportionally. Used for the sweep grid; the headline results use stride 1.
    """
    N, L = x.shape
    Xs, Ys, starts, wets = [], [], [], []
    for k in range(0, L - T - H + 1, stride):
        if keep is not None and not keep[k: k + T + H].all():
            continue
        Xs.append(x[:, k: k + T])
        Ys.append(x[:, k + T: k + T + H])
        starts.append(k)
        if wet_mask is not None:
            wets.append(bool(wet_mask[:, k + T: k + T + H].any()))
    if not Xs:
        raise ValueError("no windows survived - loosen the balancing margins")
    return (np.stack(Xs).astype("float32"), np.stack(Ys).astype("float32"),
            np.asarray(starts), np.asarray(wets, dtype=bool))


def chrono_split(n: int, frac=(0.75, 0.15, 0.10), starts=None,
                 embargo: int = 0):
    """CHRONOLOGICAL, never random. Random splits leak future into past through
    overlapping windows and inflate every number.

    With `starts` and `embargo` given, additionally drops windows whose time
    span comes within `embargo` steps of a split boundary. Consecutive sliding
    windows overlap by T+H-1 steps, so without this a test window can share
    almost all of its samples with a training window sitting just across the
    boundary - the same leak as a random split, only smaller. Pass
    embargo = T + H to guarantee no shared timestep.

    Returns three index arrays (not slices) so the embargo can remove interior
    windows.
    """
    a = int(n * frac[0])
    b = a + int(n * frac[1])
    idx = np.arange(n)
    if starts is None or embargo <= 0:
        return idx[:a], idx[a:b], idx[b:n]

    starts = np.asarray(starts)
    t_a, t_b = starts[a], starts[b] if b < n else starts[-1] + 1
    tr = idx[:a][starts[:a] + embargo <= t_a]
    va = idx[a:b][(starts[a:b] >= t_a) & (starts[a:b] + embargo <= t_b)]
    te = idx[b:n][starts[b:n] >= t_b]
    for name, part in (("train", tr), ("val", va), ("test", te)):
        if len(part) == 0:
            raise ValueError(
                f"embargo={embargo} emptied the {name} split ({n} windows). "
                "The split is narrower than the embargo, so every window in it "
                "straddles a boundary. Use a longer record or a smaller "
                "embargo - do NOT set embargo=0 to make this go away, that "
                "just restores the leak."
            )
    return tr, va, te
