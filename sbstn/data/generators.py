"""Alternative multivariate generating processes.

The advecting-cell field in `simulate.py` is one hypothesis about how spatial
structure arises across a CML network. If S-BSTN underperforms there, the
obvious objection is that the simulator - not the architecture - is at fault:
isolated Gaussian cells drifting across a sparse network may simply not contain
the kind of structure pairwise spatial attention is built to exploit.

These are genuinely different processes, added so that objection can be tested
rather than argued about:

| mode | process | what it tests |
|---|---|---|
| `var` | VAR(1) on the links, coupling from a wind-biased distance kernel | The friendliest possible case: each link IS a linear function of its neighbours' previous values. If attention cannot win here it cannot win anywhere. |
| `frontal` | a coherent line squall sweeping the domain | Long-range coherent structure rather than isolated blobs - many links wet simultaneously with a clean lead-lag ordering. |
| `multiscale` | cells spanning a wide range of size and speed | Robustness: no single characteristic length or lag for the model to latch onto. |
| `seasonal` | diurnal cycle + AR(1) noise, weak coupling | Whether the models exploit periodicity; a strong temporal-only baseline case. |

Every mode returns a rain-rate field `(N, T)` in mm/h plus a reference
influence matrix, so the rest of the pipeline - power law, observation model,
balancing, metrics, attention recovery - is unchanged.
"""
from __future__ import annotations

import numpy as np


def _geometry(links, cfg):
    mids = np.stack([np.array([(l.a[0] + l.b[0]) / 2, (l.a[1] + l.b[1]) / 2])
                     for l in links])
    th = np.deg2rad(cfg.storm_dir_deg)
    u = np.array([np.cos(th), np.sin(th)])
    diff = mids[None, :, :] - mids[:, None, :]      # diff[i, j] = m_j - m_i
    along = -(diff @ u)                              # >0 means j is upwind of i
    dist = np.linalg.norm(diff, axis=-1)
    return mids, u, along, dist


def var_process(links, cfg, rng, *, ell_frac=0.25, rho=0.92, beta=0.8):
    """VAR(1) with spatially structured, wind-biased coupling.

        s_t = A s_{t-1} + L e_t ,      r_t = relu(s_t - q) rescaled

    `A[i, j]` is large when j is close to i AND upwind of it, so the true
    Granger structure is exactly the object SCA is supposed to recover. The
    spectral radius is scaled to `rho` < 1 for stationarity.

    Returns (r_true, A) with A row-normalised for use as Lambda*.
    """
    N = len(links)
    _, _, along, dist = _geometry(links, cfg)
    ell = ell_frac * cfg.extent_km

    K = np.exp(-dist / ell)
    # Upwind sources get more weight; downwind ones are suppressed but not
    # zeroed, so the model has to learn the asymmetry rather than be handed it.
    A = K * (1.0 + beta * np.tanh(along / ell))
    np.fill_diagonal(A, 1.0)                        # self-retention
    A = np.maximum(A, 0.0)

    ev = np.max(np.abs(np.linalg.eigvals(A)))
    A = A * (rho / max(ev, 1e-9))

    n_steps = int(cfg.duration_min * 60 / cfg.dt_s)
    # Spatially correlated innovations from the same kernel.
    Sig = 0.8 * K + 0.2 * np.eye(N) + 1e-6 * np.eye(N)
    L = np.linalg.cholesky(Sig)

    s = np.zeros((N, n_steps))
    e = L @ rng.standard_normal((N, n_steps))
    for t in range(1, n_steps):
        s[:, t] = A @ s[:, t - 1] + e[:, t]

    # Rectify into a rain rate with a realistic wet fraction.
    q = np.quantile(s, 1.0 - 0.06)
    r = np.maximum(s - q, 0.0)
    r = r / max(r.max(), 1e-9) * 30.0
    return r.astype(np.float32), A


def frontal_field(links, cfg, rng, quads, counts, offsets, pts):
    """A coherent line squall sweeping across the domain.

    Unlike isolated cells, a front wets a whole swathe of the network at once
    with a clean monotone lead-lag ordering along the wind - the cleanest
    spatial signal a forecaster could hope for.
    """
    _, u, _, _ = _geometry(links, cfg)
    n_steps = int(cfg.duration_min * 60 / cfg.dt_s)
    dt_min = cfg.dt_s / 60.0
    speed = max(cfg.storm_speed_km_min, 1e-6)

    # Fronts arrive as a Poisson process, each crossing the domain once.
    span = cfg.extent_km * 2.2
    crossing = span / speed                          # minutes to traverse
    n_fronts = max(1, int(rng.poisson(cfg.spawn_per_hour * cfg.duration_min / 60.0)))
    starts = rng.uniform(-crossing, cfg.duration_min, n_fronts)
    widths = rng.uniform(1.5, 5.0, n_fronts)         # km, across-front sigma
    peaks = rng.uniform(*cfg.cell_peak_mmh, size=n_fronts)

    proj = pts @ u                                   # position along the wind
    p0 = proj.min() - span * 0.15
    out = np.zeros((n_steps, len(pts)), dtype=np.float32)
    for k in range(n_fronts):
        t_rel = (np.arange(n_steps) * dt_min - starts[k])
        live = (t_rel >= 0) & (t_rel <= crossing)
        if not live.any():
            continue
        centre = p0 + speed * t_rel[live]            # front position
        d = proj[None, :] - centre[:, None]
        out[live] += (peaks[k] * np.exp(-0.5 * (d / widths[k]) ** 2)).astype(np.float32)

    summed = np.add.reduceat(out, offsets, axis=1)
    return (summed / counts[None, :]).T.astype(np.float32)


def multiscale_cells(links, cfg, rng, quads, counts, offsets, pts):
    """Cells spanning a wide range of radius and speed, plus speed jitter.

    Removes the single characteristic length and lag that the default field
    hands the model.
    """
    n_steps = int(cfg.duration_min * 60 / cfg.dt_s)
    dt_min = cfg.dt_s / 60.0
    th = np.deg2rad(cfg.storm_dir_deg)
    base_u = np.array([np.cos(th), np.sin(th)])

    n = max(1, int(rng.poisson(cfg.spawn_per_hour * 2.5 * cfg.duration_min / 60.0)))
    t0 = rng.uniform(-90.0, cfg.duration_min, n)
    p0 = rng.uniform(-0.3, 1.3, (n, 2)) * cfg.extent_km
    # Log-uniform radii: many small cells, a few very large ones.
    sigma = np.exp(rng.uniform(np.log(1.0), np.log(12.0), n))
    peak = rng.uniform(*cfg.cell_peak_mmh, size=n)
    life = rng.uniform(15.0, 150.0, n)
    speed = cfg.storm_speed_km_min * rng.uniform(0.3, 2.0, n)
    ang = rng.normal(0.0, 0.35, n)                   # direction jitter, radians

    out = np.zeros((n_steps, len(pts)), dtype=np.float32)
    t_min = np.arange(n_steps) * dt_min
    for k in range(n):
        age = t_min - t0[k]
        live = (age >= 0) & (age <= life[k])
        if not live.any():
            continue
        c, s = np.cos(ang[k]), np.sin(ang[k])
        uk = np.array([base_u[0] * c - base_u[1] * s, base_u[0] * s + base_u[1] * c])
        env = np.sin(np.pi * np.clip(age[live] / life[k], 0, 1)) ** 0.7
        centre = p0[k][None, :] + uk[None, :] * (speed[k] * age[live])[:, None]
        d2 = ((pts[None, :, :] - centre[:, None, :]) ** 2).sum(-1)
        out[live] += (peak[k] * env)[:, None] * np.exp(-d2 / (2 * sigma[k] ** 2))

    summed = np.add.reduceat(out, offsets, axis=1)
    return (summed / counts[None, :]).T.astype(np.float32)


def seasonal_process(links, cfg, rng, *, period_min=1440.0, phi=0.995):
    """Diurnal cycle plus per-link AR(1), with only weak spatial coupling.

    A control in the other direction from `independent`: there IS strong
    structure, but it is temporal and shared, not spatial and directional.
    A purely temporal model should do well; spatial attention should add little.
    """
    N = len(links)
    n_steps = int(cfg.duration_min * 60 / cfg.dt_s)
    dt_min = cfg.dt_s / 60.0
    t = np.arange(n_steps) * dt_min

    common = np.sin(2 * np.pi * t / period_min)
    phase = rng.uniform(-0.15, 0.15, N)[:, None]     # small per-link lag
    season = np.sin(2 * np.pi * (t[None, :] / period_min) + phase)

    e = rng.standard_normal((N, n_steps))
    s = np.zeros((N, n_steps))
    for k in range(1, n_steps):
        s[:, k] = phi * s[:, k - 1] + e[:, k]
    s = s / max(s.std(), 1e-9)

    lat = 1.6 * season + 0.8 * common[None, :] + s
    q = np.quantile(lat, 1.0 - 0.06)
    r = np.maximum(lat - q, 0.0)
    return (r / max(r.max(), 1e-9) * 30.0).astype(np.float32)
