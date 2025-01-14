"""Synthetic CML network with KNOWN ground-truth spatial coupling.

The point of this module is that you choose the
storm velocity, so you know which link genuinely leads which and by how much -
which is exactly what real CML data cannot give you.

Physical chain reproduced here:

    rain field R(x, y, t)  [mm/h]
      -> path-average over each link's segment   -> r^(i)_t  [mm/h]
      -> power law                               -> gamma    [dB/km]
      -> multiply by path length                 -> A^(i)_t  [dB]
      -> baseline + wet antenna + noise          -> x^(i)_t  [dB]
      -> quantise 0.3 dB, sample 10 s            -> observed

Everything before the last two lines is ground truth and is returned alongside
the observations.

Implementation note: the field is evaluated in VECTORISED TIME CHUNKS. The
obvious per-timestep Python loop costs ~87 s for a 30-day N=16 scenario, which
makes parameter sweeps impractical. Cells are pre-generated
as a Poisson process over the whole duration, then each chunk evaluates only
the cells alive within it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

from . import generators as G


# ----------------------------------------------------------------------
# Geometry
@dataclass
class Link:
    id: int
    a: tuple[float, float]          # endpoint A, km
    b: tuple[float, float]          # endpoint B, km
    freq_ghz: float
    polarization: str = "V"
    antenna_diam_m: float = 3.0

    @property
    def length_km(self) -> float:
        return float(np.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1]))

    @property
    def midpoint(self) -> np.ndarray:
        return np.array([(self.a[0] + self.b[0]) / 2, (self.a[1] + self.b[1]) / 2])

    def quad_points(self, n: int | None = None) -> np.ndarray:
        """Discretise the path for the path-average integral (path-averaged rain rate)."""
        n = n or max(32, int(self.length_km * 10))
        s = np.linspace(0.0, 1.0, n)[:, None]
        return np.array(self.a)[None] * (1 - s) + np.array(self.b)[None] * s


def _freq_for(length_km: float, rng) -> float:
    """Short hops run higher frequency in real backhaul. Loosely mimic that."""
    f = 40.0 - 2.5 * length_km + rng.normal(0, 2.0)
    return float(np.clip(f, 18.0, 40.0))


def make_chain(N: int, extent: float, rng) -> list[Link]:
    """Linear chain, D3-like. Nodes strung across the domain."""
    xs = np.linspace(0.1 * extent, 0.9 * extent, N + 1)
    ys = 0.5 * extent + rng.normal(0, 0.03 * extent, N + 1)
    links = []
    for i in range(N):
        a, b = (xs[i], ys[i]), (xs[i + 1], ys[i + 1])
        L = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        links.append(Link(i, a, b, _freq_for(L, rng)))
    return links


def make_ring(N: int, extent: float, rng) -> list[Link]:
    """Closed ring, D2-like. Tight by design - every link sees the same weather,
    which is the condition the paper says defeats SAN."""
    r = 0.22 * extent
    cx = cy = extent / 2
    th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], -1)
    links = []
    for i in range(N):
        a, b = tuple(pts[i]), tuple(pts[(i + 1) % N])
        L = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        links.append(Link(i, a, b, _freq_for(L, rng)))
    return links


def make_tree(N: int, extent: float, rng) -> list[Link]:
    """Hub and spoke with chained branches, D1-like."""
    hub = np.array([extent / 2, extent / 2])
    links, nodes, i = [], [hub], 0
    while i < N:
        parent = nodes[rng.integers(0, len(nodes))]
        ang = rng.uniform(0, 2 * np.pi)
        L = rng.uniform(1.0, 0.18 * extent)
        child = parent + L * np.array([np.cos(ang), np.sin(ang)])
        child = np.clip(child, 0, extent)
        links.append(Link(i, tuple(parent), tuple(child), _freq_for(L, rng)))
        nodes.append(child)
        i += 1
    return links


def make_random(N: int, extent: float, rng) -> list[Link]:
    links = []
    for i in range(N):
        a = rng.uniform(0, extent, 2)
        ang = rng.uniform(0, 2 * np.pi)
        L = rng.uniform(1.0, 8.0)
        b = np.clip(a + L * np.array([np.cos(ang), np.sin(ang)]), 0, extent)
        links.append(Link(i, tuple(a), tuple(b), _freq_for(L, rng)))
    return links


TOPOLOGIES = {"chain": make_chain, "ring": make_ring,
              "tree": make_tree, "random": make_random}


# ----------------------------------------------------------------------
@dataclass
class SimConfig:
    n_links: int = 16
    topology: str = "chain"
    extent_km: float = 30.0

    # advect | uniform | independent | static | var | frontal | multiscale | seasonal
    # The last four live in generators.py - alternative multivariate processes,
    # added so "the simulator is at fault" can be tested rather than argued.
    field_mode: str = "advect"
    storm_speed_km_min: float = 0.5     # 30 km/h
    storm_dir_deg: float = 0.0          # 0 = +x (east)
    cell_sigma_km: tuple[float, float] = (3.0, 8.0)
    cell_peak_mmh: tuple[float, float] = (5.0, 40.0)
    cell_life_min: tuple[float, float] = (20.0, 90.0)

    # CALIBRATED to the paper's raw wet fraction.
    #
    # The paper's RAW wet fraction is 1.8-2.3% and only reaches 33-44% AFTER
    # the paper's wet-dry balancing. At spawn_per_hour=2.0 the expected
    # number of concurrent cells is 2.0 * (55 min mean life / 60) = 1.8, each
    # 3-8 km wide on a 30 km domain, which rains ~35% of the time. Balancing a
    # 35%-wet record is a no-op, so the pipeline the paper spends a section on
    # would never be exercised.
    #
    # 0.08/h was chosen by sweeping the rate over a 30-day N=16 chain scenario
    # and reading off the raw wet fraction: it gives 2.00%, inside the paper's
    # 1.8-2.3% band, with a mean wet-period rain rate of 2.35 mm/h against the
    # paper's 2.6 mm/h. calibrate_spawn_rate() re-derives it for other configs.
    spawn_per_hour: float = 0.08

    duration_min: float = 60 * 24 * 30  # 30 days
    dt_s: float = 10.0                  # paper sampling interval

    # observation model
    baseline_dB: tuple[float, float] = (30.0, 60.0)
    baseline_drift_dB: float = 0.2
    wet_antenna_gain_dB: float = 1.5
    wet_antenna_tau_min: float = 10.0
    noise_sigma_dB: float = 0.1
    quantization_dB: float = 0.3        # paper value, keep it

    # PLACEHOLDER power law. NOT ITU-R P.838-3. Frequency-independent, so it
    # cannot reproduce the frequency-dependent sensitivity that motivates the
    # paper. The real coefficient tables must be taken FROM THE
    # RECOMMENDATION, so they are NOT bundled here.
    # See itu_r_838.py for the loader and what it requires.
    #
    # Consequence, stated plainly: every rain-rate number this simulator
    # produces is internally consistent but physically unanchored. The forward
    # law and its inverse agree exactly, so the IDW rain-map module can be
    # validated as a PIPELINE, but no quantitative mm/h claim is supportable
    # until the real tables are in place.
    pl_k: float = 0.1
    pl_alpha: float = 1.0

    # Forecast horizon in MINUTES, used only to build the reference influence
    # matrix Lambda*. Default 5.0 = H(10) * 30 s, the paper's horizon.
    horizon_min: float = 5.0

    # Time-chunk size for the vectorised field evaluation. Memory is
    # O(chunk * total_quad_points).
    chunk_steps: int = 4096

    seed_geometry: int = 0
    seed_field: int = 1
    seed_noise: int = 2


# ----------------------------------------------------------------------
class RainField:
    """Superposed advecting Gaussian cells, evaluated in vectorised time chunks.

    The SHARED velocity vector is the whole point: it creates deterministic
    lead-lag between links, which is the structure spatial attention is meant
    to discover and which you know the answer to because you chose it.

    Cells are pre-generated as a Poisson process over the whole duration rather
    than sampled step by step, so a chunk can be evaluated with no Python loop
    over time.
    """

    def __init__(self, cfg: SimConfig, rng):
        self.cfg = cfg
        th = np.deg2rad(cfg.storm_dir_deg)
        self.v = cfg.storm_speed_km_min * np.array([np.cos(th), np.sin(th)])
        if cfg.field_mode == "static":
            self.v = np.zeros(2)

        n_exp = cfg.spawn_per_hour * cfg.duration_min / 60.0
        n = int(rng.poisson(max(n_exp, 0.0)))
        # Spawn slightly before t=0 too, so the record does not begin in an
        # artificially rain-free state.
        lead = float(cfg.cell_life_min[1])
        self.t0 = np.sort(rng.uniform(-lead, cfg.duration_min, n))
        self.p0 = rng.uniform(-0.3, 1.3, (n, 2)) * cfg.extent_km
        self.sigma = rng.uniform(*cfg.cell_sigma_km, size=n)
        self.peak = rng.uniform(*cfg.cell_peak_mmh, size=n)
        self.life = rng.uniform(*cfg.cell_life_min, size=n)
        self.n_cells = n

    def _intensity(self, age: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Ramp up, plateau, decay. age: (K, C) minutes since spawn."""
        life = self.life[idx][None, :]
        f = np.clip(age / life, 0.0, 1.0)
        env = np.sin(np.pi * f) ** 0.7
        env = np.where((age >= 0) & (age <= life), env, 0.0)
        return self.peak[idx][None, :] * env                      # (K, C)

    def at_times(self, pts: np.ndarray, t_mins: np.ndarray) -> np.ndarray:
        """Rain rate [mm/h] at (P, 2) points for (K,) times -> (K, P)."""
        c = self.cfg
        K, P = len(t_mins), len(pts)
        if c.field_mode == "independent" or self.n_cells == 0:
            return np.zeros((K, P), dtype=np.float32)

        # Only cells whose lifetime overlaps this chunk.
        lo, hi = float(t_mins[0]), float(t_mins[-1])
        idx = np.flatnonzero((self.t0 <= hi) & (self.t0 + self.life >= lo))
        if idx.size == 0:
            return np.zeros((K, P), dtype=np.float32)

        age = t_mins[:, None] - self.t0[idx][None, :]             # (K, C)
        I = self._intensity(age, idx)                             # (K, C)

        if c.field_mode == "uniform":
            # Same everywhere: no spatial structure at all.
            return np.repeat(I.sum(1)[:, None], P, axis=1).astype(np.float32)

        out = np.zeros((K, P), dtype=np.float64)
        sig2 = 2.0 * self.sigma[idx] ** 2                         # (C,)
        for m in range(idx.size):
            Im = I[:, m]
            live = Im > 0
            if not live.any():
                continue
            centre = self.p0[idx[m]][None, :] + self.v[None, :] * age[live, m][:, None]
            d2 = ((pts[None, :, :] - centre[:, None, :]) ** 2).sum(-1)   # (k, P)
            out[live] += Im[live, None] * np.exp(-d2 / sig2[m])
        return out.astype(np.float32)


# ----------------------------------------------------------------------
def reference_influence(links, cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth influence matrix Lambda* and lead-lag matrix.

    Link j predicts link i to the degree that j sits UPWIND of i, is laterally
    aligned with it, and leads it by a lag inside the forecast horizon.
    """
    N = len(links)
    th = np.deg2rad(cfg.storm_dir_deg)
    v = cfg.storm_speed_km_min * np.array([np.cos(th), np.sin(th)])
    speed = float(np.linalg.norm(v))
    if cfg.field_mode == "static":
        speed = 0.0
    u = v / speed if speed > 0 else np.zeros(2)

    mids = np.stack([l.midpoint for l in links])
    d = mids[None, :, :] - mids[:, None, :]              # d[i, j] = m_j - m_i
    # along[i, j] > 0 means j sits UPWIND of i: the storm moves along +u, so
    # it reaches j first and j LEADS i by along / speed minutes.
    along = -(d @ u)                                     # (N, N)
    across = np.linalg.norm(d + along[..., None] * u[None, None, :], axis=-1)
    sigma = float(np.mean(cfg.cell_sigma_km))

    lag = along / speed if speed > 0 else np.zeros((N, N))   # lead of j over i
    lam = np.exp(-(across ** 2) / (2 * sigma ** 2)) * np.exp(-np.abs(along) / (3 * sigma))
    if speed > 0:
        #     Lambda*_ij  ~  exp(-across^2/2sigma^2)
        #                    * 1[ 0 <= lag*_ij <= H*dt ]
        #                    * exp(-along/lambda_decay)
        #
        # The indicator is not cosmetic. A link 16 km upwind leads its target
        # by 32 min at 0.5 km/min; it cannot inform a 5-minute forecast. j is
        # useful only when it is upwind AND leads by a lag that falls inside
        # the horizon - the temporal scale of the task bounds the spatial one.
        lam = np.where((along < 0) | (lag > cfg.horizon_min), 0.0, lam)
    dead = lam.sum(1) <= 0
    if dead.any():
        lam[dead] = np.eye(N)[dead]
    lam = lam / lam.sum(1, keepdims=True)
    return lam, lag


def calibrate_spawn_rate(cfg: SimConfig, target=0.02, days=4.0,
                         lo=0.01, hi=3.0, iters=12) -> float:
    """Bisect spawn_per_hour so the RAW wet fraction lands near `target`.

    The paper's raw wet fraction is 1.8-2.3%; balancing then lifts it to
    33-44%. Calibrating the raw rate is what makes the balancing pipeline a
    real step rather than a no-op. Used to choose SimConfig.spawn_per_hour.
    """
    from dataclasses import replace
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        probe = replace(cfg, spawn_per_hour=mid, duration_min=60 * 24 * days)
        frac = float(simulate(probe, observations=False)["wet_mask"].mean())
        if frac > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def simulate(cfg: SimConfig, observations: bool = True) -> dict:
    """Generate one scenario. Returns arrays + ground truth, all saveable.

    `observations=False` skips the observation model and returns only the rain
    ground truth - used by calibrate_spawn_rate, which only needs wet_mask.
    """
    g_rng = np.random.default_rng(cfg.seed_geometry)
    f_rng = np.random.default_rng(cfg.seed_field)
    n_rng = np.random.default_rng(cfg.seed_noise)

    links = TOPOLOGIES[cfg.topology](cfg.n_links, cfg.extent_km, g_rng)
    N = len(links)
    n_steps = int(cfg.duration_min * 60 / cfg.dt_s)
    dt_min = cfg.dt_s / 60.0
    lengths = np.array([l.length_km for l in links])

    r_true = np.zeros((N, n_steps), dtype=np.float32)

    var_A = None
    if cfg.field_mode in ("var", "seasonal"):
        if cfg.field_mode == "var":
            r_true, var_A = G.var_process(links, cfg, f_rng)
        else:
            r_true = G.seasonal_process(links, cfg, f_rng)
    elif cfg.field_mode in ("frontal", "multiscale"):
        quads = [l.quad_points() for l in links]
        counts = np.array([len(q) for q in quads])
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
        pts = np.concatenate(quads, axis=0)
        fn = G.frontal_field if cfg.field_mode == "frontal" else G.multiscale_cells
        r_true = fn(links, cfg, f_rng, quads, counts, offsets, pts)
    elif cfg.field_mode == "independent":
        # Per-link AR(1) bursts, NO shared field. Spatial attention must give
        # zero benefit here. If it helps, you have a leak. Run this FIRST.
        phi = 0.995
        e = f_rng.standard_normal((N, n_steps))
        s = np.zeros((N, n_steps))
        for t in range(1, n_steps):                     # AR(1) is inherently serial
            s[:, t] = phi * s[:, t - 1] + e[:, t]
        thr = np.quantile(s, 0.97, axis=1, keepdims=True)
        s = np.maximum(s - thr, 0.0)
        r_true = (s / np.maximum(s.max(axis=1, keepdims=True), 1e-6) * 25.0).astype(np.float32)
    else:
        fieldgen = RainField(cfg, f_rng)
        # One flat array of every link's quadrature points, plus the offsets
        # that let np.add.reduceat path-average each link in one shot.
        quads = [l.quad_points() for l in links]
        counts = np.array([len(q) for q in quads])
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
        pts = np.concatenate(quads, axis=0)             # (P, 2)

        for lo in range(0, n_steps, cfg.chunk_steps):
            hi = min(lo + cfg.chunk_steps, n_steps)
            t_mins = np.arange(lo, hi) * dt_min
            field = fieldgen.at_times(pts, t_mins)      # (K, P)
            summed = np.add.reduceat(field, offsets, axis=1)    # (K, N)
            r_true[:, lo:hi] = (summed / counts[None, :]).T.astype(np.float32)

    # power law -> attenuation
    gamma = cfg.pl_k * np.power(np.maximum(r_true, 0.0), cfg.pl_alpha)
    A = (gamma * lengths[:, None]).astype(np.float32)
    wet_mask = r_true > 0.1

    if not observations:
        return dict(r_true=r_true, A_clean=A, wet_mask=wet_mask,
                    links=[asdict(l) for l in links], config=asdict(cfg))

    base = n_rng.uniform(*cfg.baseline_dB, size=(N, 1))
    drift = np.cumsum(
        n_rng.normal(0, cfg.baseline_drift_dB / np.sqrt(n_steps), (N, n_steps)), axis=1
    )

    # Wet antenna: rises during rain, decays with a time constant after it
    # stops. The paper extends its training windows past the end of rain
    # specifically to capture this, so a simulator without it cannot exercise
    # that part of the pipeline.
    decay = np.exp(-dt_min / cfg.wet_antenna_tau_min)
    W = np.zeros_like(A)
    target = wet_mask.astype(np.float32) * cfg.wet_antenna_gain_dB
    for t in range(1, n_steps):
        W[:, t] = np.maximum(target[:, t], W[:, t - 1] * decay)

    noise = n_rng.normal(0, cfg.noise_sigma_dB, (N, n_steps))
    x = A + base + drift + W + noise
    # quantise LAST, after every additive term
    x = np.round(x / cfg.quantization_dB) * cfg.quantization_dB

    lam_star, lag_star = reference_influence(links, cfg)
    if var_A is not None:
        # The VAR coefficient matrix is the ground-truth influence by
        # construction, so use it directly rather than the geometric proxy.
        # lag* stays geometric, which keeps `upwind mass` a statement about
        # physical lead-lag rather than a tautology about the lag-1 structure.
        lam_star = (var_A / np.maximum(var_A.sum(1, keepdims=True), 1e-9)).astype(np.float32)

    return dict(
        x_obs=x.astype(np.float32),
        r_true=r_true,
        A_clean=A,
        wet_mask=wet_mask,
        lambda_star=lam_star.astype(np.float32),
        lag_star=lag_star.astype(np.float32),
        links=[asdict(l) for l in links],
        config=asdict(cfg),
    )


def empirical_influence(r_true: np.ndarray, max_lag: int, max_steps: int = 60000) -> np.ndarray:
    """Lagged cross-correlation version of Lambda*.

    Must agree with reference_influence(). If it does not, the field generator
    and the geometry disagree and something is wrong BEFORE any model trains.

    out[i, j] = max over lag L >= 0 of corr( r_i(t), r_j(t - L) ), i.e. how well
    source j LEADS target i. Long records are subsampled to bound the cost,
    which is O(N^2 * max_lag * T).
    """
    N, T = r_true.shape
    if T > max_steps:
        r_true = r_true[:, :max_steps]
        T = max_steps
    z = (r_true - r_true.mean(1, keepdims=True)) / (r_true.std(1, keepdims=True) + 1e-9)
    out = np.zeros((N, N))
    for L in range(0, max_lag + 1):
        a = z[:, L:] if L else z
        b = z[:, :T - L] if L else z
        c = (a @ b.T) / a.shape[1]                       # c[i, j] = corr(z_i(t), z_j(t-L))
        out = np.maximum(out, c)
    out = np.maximum(out, 0.0)
    return out / out.sum(1, keepdims=True).clip(1e-9)
