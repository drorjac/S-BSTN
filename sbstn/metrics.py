"""Evaluation metrics (paper, "Evaluation Metrics").

ALL METRICS IN dB - invert the min-max scaling before calling these. Values
computed on the [0,1] scaled signal are off by the scale factor and comparable
to nothing in the paper; the tell-tale symptom is metrics that look tiny.
"""
from __future__ import annotations
import numpy as np


def per_horizon(y: np.ndarray, yhat: np.ndarray) -> dict:
    """y, yhat: (S, N, H) in dB -> dict of (H,) arrays.

    `avg` aggregates over all links; `max` isolates the worst link per time
    step. The point of `max` is that when rain covers only part of the
    subnetwork, `avg` is diluted by the dry links.
    """
    e = y - yhat
    return dict(
        rmse_avg=np.sqrt((e ** 2).mean(axis=1).mean(axis=0)),
        rmse_max=np.sqrt((e ** 2).max(axis=1).mean(axis=0)),
        mae_avg=np.abs(e).mean(axis=1).mean(axis=0),
        mae_max=np.abs(e).max(axis=1).mean(axis=0),
    )


def tnerror(m: dict) -> dict:
    """Window-averaged score: mean over horizon steps. TNERROR in the paper."""
    return {k: float(v.mean()) for k, v in m.items()}


def report(y, yhat, steps=(2, 6, 10), dt_min=0.5):
    """Per-horizon table at the paper's 1 / 3 / 5 min marks, plus TNERROR."""
    if len(y) == 0:
        return {}
    m = per_horizon(y, yhat)
    rows = {}
    H = len(next(iter(m.values())))
    for h in steps:
        if h <= H:
            rows[f"{h * dt_min:g}min"] = {k: float(v[h - 1]) for k, v in m.items()}
    rows["TN"] = tnerror(m)
    return rows


def split_wet_dry(y, yhat, wet_flag, **kw):
    """Report wet and dry separately. Pooled averages hide everything: dry
    error sits below the 0.3 dB quantisation for every model including naive."""
    wet_flag = np.asarray(wet_flag, dtype=bool)
    return dict(
        all=report(y, yhat, **kw),
        wet=report(y[wet_flag], yhat[wet_flag], **kw),
        dry=report(y[~wet_flag], yhat[~wet_flag], **kw),
        n_wet=int(wet_flag.sum()), n_dry=int((~wet_flag).sum()),
    )


# ---- attention recovery against the simulator's ground truth ----
def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r that returns nan rather than raising on a constant row.

    A row of Lambda can legitimately be constant - a fully pruned row falls
    back to one-hot, and a dense row at initialisation is near-uniform - and
    np.corrcoef emits a 0/0 warning and a nan for those. Those rows carry no
    rank information, so they are dropped from the average rather than being
    allowed to poison it.
    """
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def attention_recovery(lam_learned: np.ndarray, lam_star: np.ndarray,
                       lag_star: np.ndarray, k: int = 3,
                       horizon_min: float | None = None) -> dict:
    """lam_learned: (N, N) time-averaged.

    Upwind mass is the headline number - a model that learned the physics puts
    its mass upwind. Chance is ~0.5.
    """
    N = lam_star.shape[0]
    kk = int(min(k, N))
    top_l = np.argsort(-lam_learned, axis=1)[:, :kk]
    top_s = np.argsort(-lam_star, axis=1)[:, :kk]
    overlap = np.mean([len(set(top_l[i]) & set(top_s[i])) / kk for i in range(N)])

    # "Upwind" means j leads i by a POSITIVE lag, and NOTHING else - that is
    # the definition used here and the reason its chance level is ~0.5 (by
    # symmetry, half of all ordered pairs are upwind).
    #
    # Do NOT fold the horizon restriction into this number. Requiring
    # 0 < lag <= horizon as well leaves only a small minority of pairs
    # eligible, which drags the chance level down to a few percent and makes a
    # perfectly ordinary model look like a total failure - 0.057 against an
    # assumed chance of 0.5. That restricted quantity is still worth having, so
    # it is reported separately, next to the chance level it should be judged
    # against.
    total = max(lam_learned.sum(), 1e-9)
    upwind = float((lam_learned * (lag_star > 0)).sum() / total)
    out_horizon = {}
    if horizon_min is not None:
        inside = (lag_star > 0) & (lag_star <= horizon_min)
        out_horizon = dict(
            in_horizon_mass=float((lam_learned * inside).sum() / total),
            in_horizon_chance=float(inside.mean()),
        )

    corrs = [_safe_corr(lam_learned[i], lam_star[i]) for i in range(N)]
    corrs = [c for c in corrs if np.isfinite(c)]
    return dict(
        topk_overlap=float(overlap),
        upwind_mass=upwind,               # chance ~0.5
        upwind_chance=float((lag_star > 0).mean()),
        row_corr=float(np.mean(corrs)) if corrs else float("nan"),
        sparsity=float((lam_learned > 1e-6).mean()),
        **out_horizon,
    )


def skill_vs(reference: dict, model: dict, key="rmse_avg", split="wet") -> float:
    """Percentage improvement of `model` over `reference` on TNERROR.

    Positive means the model is better. This is the quantity the paper reports
    as its 17-20% margin.
    """
    r = reference[split]["TN"][key]
    m = model[split]["TN"][key]
    return 100.0 * (r - m) / max(r, 1e-12)
