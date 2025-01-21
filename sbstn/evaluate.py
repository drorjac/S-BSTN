"""Downstream evaluation: attenuation -> rain rate -> interpolated rain map.

Not part of the network; it is how the paper
motivates the work, and it is built as a separate module so the forecasting
results do not depend on it.

    1. predict the attenuation window for all links
    2. convert each link to a path-averaged rain rate via the power law
    3. interpolate to a grid by inverse distance weighting

IMPORTANT CAVEAT, and it is not a small one: step 2 uses the simulator's
PLACEHOLDER power law (k = 0.1, alpha = 1.0, frequency-independent) unless
the ITU-R P.838-3 coefficient table has been supplied (see data/itu_r_838.py). The forward law
and this inverse are exact mutual inverses, so what follows validates the
PIPELINE - that a predicted attenuation field becomes a sensible rain map -
and not the physics. No mm/h number here is physically anchored. See
data/itu_r_838.py.
"""
from __future__ import annotations

import numpy as np

from .data.itu_r_838 import coefficients_available, k_alpha


def link_rain_rate(A_dB, lengths_km, k=0.1, alpha=1.0):
    """Invert gamma = k R^alpha over each link path. A_dB: (..., N)."""
    gamma = np.maximum(np.asarray(A_dB, float), 0.0) / np.maximum(lengths_km, 1e-9)
    return np.power(gamma / k, 1.0 / alpha)


def coefficients_for(links, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Real P.838-3 coefficients when the table is present, placeholder else."""
    if coefficients_available():
        f = np.array([l["freq_ghz"] for l in links])
        pol = links[0].get("polarization", "V")
        return k_alpha(f, pol)
    n = len(links)
    return np.full(n, cfg["pl_k"]), np.full(n, cfg["pl_alpha"])


def idw_map(values, points, grid_x, grid_y, power=2.0, eps=1e-6):
    """Inverse-distance-weighted interpolation onto a grid.

        r^g(x,y) = SUM_i r_i w_i / SUM_i w_i ,   w_i = 1 / d_i^p

    values: (N,) per-link rain rate, points: (N, 2) link midpoints.
    Returns (len(grid_y), len(grid_x)).
    """
    gx, gy = np.meshgrid(grid_x, grid_y)
    d = np.sqrt((gx[..., None] - points[:, 0]) ** 2 +
                (gy[..., None] - points[:, 1]) ** 2)
    w = 1.0 / np.power(np.maximum(d, eps), power)
    return (w * values).sum(-1) / w.sum(-1)


def rain_map_scores(y_db, pred_db, links, cfg, extent_km, n_grid=48, power=2.0):
    """Compare the rain map built from predicted vs true attenuation.

    y_db, pred_db: (S, N, H) in dB. Returns per-horizon rain-domain scores plus
    the arrays needed to draw the maps.
    """
    # Link is a dataclass whose length_km / midpoint are PROPERTIES, and
    # asdict() does not serialise properties - so recompute both from the
    # endpoints rather than expecting them in the saved metadata.
    lengths = np.array([float(np.hypot(l["b"][0] - l["a"][0], l["b"][1] - l["a"][1]))
                        for l in links])
    k, alpha = coefficients_for(links, cfg)
    mids = np.array([[(l["a"][0] + l["b"][0]) / 2, (l["a"][1] + l["b"][1]) / 2]
                     for l in links])

    # Attenuation here is the observed signal, which includes the baseline. The
    # rain-bearing part is the excursion above each link's dry level, estimated
    # as a low quantile over the evaluation set - the same idea as the paper's
    # baseline removal, done simply because the simulator's baseline is slow.
    base = np.quantile(y_db, 0.05, axis=(0, 2), keepdims=True)
    r_true = link_rain_rate(y_db - base, lengths[None, :, None], k[None, :, None],
                            alpha[None, :, None])
    r_pred = link_rain_rate(pred_db - base, lengths[None, :, None], k[None, :, None],
                            alpha[None, :, None])

    g = np.linspace(0, extent_km, n_grid)
    per_h = []
    for h in range(y_db.shape[-1]):
        mt = np.stack([idw_map(r_true[s, :, h], mids, g, g, power)
                       for s in range(len(y_db))])
        mp = np.stack([idw_map(r_pred[s, :, h], mids, g, g, power)
                       for s in range(len(y_db))])
        e = mt - mp
        per_h.append(dict(
            rmse_mmh=float(np.sqrt((e ** 2).mean())),
            bias_mmh=float(e.mean()),
            corr=float(np.corrcoef(mt.ravel(), mp.ravel())[0, 1]),
        ))
    return dict(
        per_horizon=per_h,
        link_rmse_mmh=float(np.sqrt(((r_true - r_pred) ** 2).mean())),
        placeholder_power_law=not coefficients_available(),
        grid=g, mids=mids, r_true=r_true, r_pred=r_pred,
    )
