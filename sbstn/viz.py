"""Visualisation for S-BSTN: the network, the physics, and what the model learned.

Every function takes plain arrays / result dicts, returns a matplotlib Figure,
and saves it when `path` is given. Nothing here needs a GPU or the model class.

    network + storm                plot_network, plot_event
    spatio-temporal physics        plot_spatiotemporal_structure
    spatial attention  (Lambda)    plot_spatial_attention, plot_attention_matrices
    temporal attention (pi)        plot_temporal_attention
    forecasts & skill              plot_forecast, plot_horizon, plot_ablation
    training dynamics              plot_training
    downstream rain map            plot_rain_map

Colour is assigned by role in a fixed order and never cycled; sequential maps
use one hue, diverging maps a neutral midpoint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
    try:  # headless by default; notebooks keep their inline backend
        matplotlib.use("Agg")
    except Exception:  # pragma: no cover
        pass
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# ---- design tokens -------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"
LINK = "#b9c3cf"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Colour follows the model, not its rank, so each arm keeps its hue everywhere.
ARM_COLOR = {
    "S-BSTN": SERIES[0], "STANN": SERIES[1], "LSTM-ED": SERIES[2],
    "persistence": INK_MUTED,
    "S-BSTN-V1": SERIES[3], "S-BSTN-V2": SERIES[4],
    "S-BSTN-V3": SERIES[5], "S-BSTN-V4": SERIES[6],
}
ABLATION_LABEL = {
    "S-BSTN": "S-BSTN (full)",
    "S-BSTN-V1": "V1  − spatial attention",
    "S-BSTN-V2": "V2  − temporal attention",
    "S-BSTN-V3": "V3  − SAN pruning",
    "S-BSTN-V4": "V4  − bidirectionality",
    "STANN": "STANN",
    "LSTM-ED": "LSTM-ED",
    "persistence": "persistence",
}

SEQ = LinearSegmentedColormap.from_list("sbstn_seq", ["#f4f8fd", "#2a78d6", "#123a6b"])
DIV = LinearSegmentedColormap.from_list("sbstn_div", ["#1b6ec2", "#eeeeec", "#d1502a"])


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2, "axes.titlesize": 10,
        "axes.titleweight": "semibold", "axes.titlecolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "figure.dpi": 150,
    })


def _ax(ax, xlabel=None, ylabel=None, title=None, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis=grid, alpha=0.9)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", pad=8)
    return ax


def _finish(fig, path):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig


def _colorbar(fig, im, ax, label=None):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2)
    if label:
        cb.set_label(label, fontsize=8)
    return cb


def _mids(links) -> np.ndarray:
    return np.array([[(l["a"][0] + l["b"][0]) / 2, (l["a"][1] + l["b"][1]) / 2]
                     for l in links])


def _unit(deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    return np.array([np.cos(th), np.sin(th)])


def _draw_links(ax, links, color=LINK, lw=2.2, label_ids=False):
    for l in links:
        ax.plot([l["a"][0], l["b"][0]], [l["a"][1], l["b"][1]], color=color, lw=lw,
                solid_capstyle="round", zorder=1)
    m = _mids(links)
    ax.scatter(m[:, 0], m[:, 1], s=9, color=INK_2, zorder=3, linewidths=0)
    if label_ids:
        for l, (x, y) in zip(links, m):
            ax.annotate(str(l["id"]), (x, y), xytext=(3, 3), textcoords="offset points",
                        fontsize=6.5, color=INK_2)
    ax.set_aspect("equal")


def _storm_arrow(ax, extent, storm_dir_deg, label="storm motion"):
    u = _unit(storm_dir_deg)
    o = np.array([0.08, 0.9]) * extent
    ax.annotate("", xy=o + 0.2 * extent * u, xytext=o,
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.6))
    ax.text(o[0], o[1] + 0.05 * extent, label, color=INK_2, fontsize=8)


# ---- network & physics -----------------------------------------------------
def plot_network(links, extent_km=None, storm_dir_deg=None, title=None, path=None,
                 ax=None):
    """CML geometry: each link is the segment it path-averages the rain over."""
    style()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 4.0))
    _draw_links(ax, links, color=SERIES[0], label_ids=True)
    if extent_km is None:
        pts = np.array([p for l in links for p in (l["a"], l["b"])])
        extent_km = float(pts.max())
    if storm_dir_deg is not None:
        _storm_arrow(ax, extent_km, storm_dir_deg)
    ax.set_xlim(-0.02 * extent_km, 1.02 * extent_km)
    ax.set_ylim(-0.02 * extent_km, 1.02 * extent_km)
    _ax(ax, "x [km]", "y [km]", title or f"CML network, N = {len(links)}", grid=None)
    return _finish(fig or ax.figure, path)


def _pick_event(r, along, half, n_show, n_candidates=10, min_frac=0.2):
    """Index of the event whose lead-lag is easiest to see.

    Candidates are the strongest well-separated rain peaks (network-total rain
    at least `min_frac` of the strongest). Among those whose most-hit links
    peak in upwind -> downwind order, take the one with the widest spread of
    peak times.
    """
    tot = np.convolve(r.sum(0), np.ones(30) / 30, mode="same")
    peaks = []
    for p in np.argsort(-tot):
        if tot[p] <= 0 or len(peaks) >= n_candidates:
            break
        if all(abs(p - q) > 2 * half for q in peaks):
            peaks.append(int(p))
    best, best_spread = peaks[0], -1.0
    for p in peaks:
        if tot[p] < min_frac * tot[peaks[0]]:
            continue
        seg = r[:, max(0, p - half):p + half]
        hit = sorted(np.argsort(-seg.max(1))[:n_show], key=lambda i: along[i])
        tp = np.array([seg[i].argmax() for i in hit], dtype=float)
        if np.all(np.diff(tp) >= 0) and tp[-1] - tp[0] > best_spread:
            best, best_spread = p, tp[-1] - tp[0]
    return best


def plot_event(sim, path=None, event_idx=None, n_show=4):
    """A rain event crossing the network: geometry + the signals it leaves.

    The links shown are ordered along the storm direction and drawn as excess
    attenuation over their own dry level, so the lead-lag that spatial
    attention has to discover is visible directly in the traces. By default
    the event with the clearest lead-lag is chosen (see _pick_event).
    """
    style()
    cfg = sim["config"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7),
                             gridspec_kw={"width_ratios": [1, 1.55]})
    links = sim["links"]
    plot_network(links, cfg["extent_km"], cfg["storm_dir_deg"],
                 title=f"(a) {cfg['topology']} network, N = {len(links)}", ax=axes[0])

    x, r = sim["x_obs"], sim["r_true"]
    half = int(45 * 60 / cfg["dt_s"])
    along = _mids(links) @ _unit(cfg["storm_dir_deg"])
    if event_idx is None:
        event_idx = _pick_event(r, along, half, n_show)
    lo, hi = max(0, event_idx - half), min(x.shape[1], event_idx + half)
    t = (np.arange(lo, hi) - event_idx) * cfg["dt_s"] / 60.0
    hit = np.argsort(-r[:, lo:hi].max(1))[:n_show]
    hit = sorted(hit, key=lambda i: along[i])
    ax = axes[1]
    for k, i in enumerate(hit):
        xi = x[i, lo:hi]
        ax.plot(t, xi - np.percentile(xi, 10), color=SERIES[k], lw=1.5, label=f"link {i}")
        axes[0].plot(*zip(links[i]["a"], links[i]["b"]), color=SERIES[k], lw=3.2, zorder=2)
    ax.legend(ncol=n_show, loc="upper left", bbox_to_anchor=(0, 1.02))
    _ax(ax, "time relative to event peak [min]", "attenuation above dry level [dB]",
        "(b) observed signals, ordered upwind → downwind")
    fig.tight_layout()
    return _finish(fig, path)


def _lagged_xcorr(r, max_lag):
    """peak[i, j], lag[i, j]: best corr of r_i(t) with r_j(t - L), L in [-max, max]."""
    N, T = r.shape
    z = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-9)
    best = np.full((N, N), -np.inf)
    arg = np.zeros((N, N))
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            a, b = z[:, L:], z[:, :T - L]
        else:
            a, b = z[:, :T + L], z[:, -L:]
        c = (a @ b.T) / a.shape[1]
        upd = c > best
        best = np.where(upd, c, best)
        arg = np.where(upd, L, arg)
    return best, arg


def plot_spatiotemporal_structure(sim, path=None, max_lag_min=30.0, max_steps=200_000):
    """The physics S-BSTN is built around, measured from the rain field itself.

    (a) Temporal scale: the lag at which link j best predicts link i grows
        linearly with their along-storm separation, slope 1 / storm speed.
        A source is only useful to a forecast if that lag fits the horizon.
    (b) Spatial scale: peak correlation decays with link separation on the
        rain-cell length scale. Beyond it a link carries no information
        about the target, which is exactly what SAN pruning removes.
    """
    style()
    cfg = sim["config"]
    dt_min = cfg["dt_s"] / 60.0
    r = sim["r_true"][:, :max_steps].astype(np.float64)
    wet_any = r.max(0) > 0.1
    r = r[:, wet_any] if wet_any.sum() > 100 else r
    peak, lag = _lagged_xcorr(r, int(max_lag_min / dt_min))
    lag_min = lag * dt_min

    m = _mids(sim["links"])
    u = _unit(cfg["storm_dir_deg"])
    d = m[None, :, :] - m[:, None, :]              # d[i, j] = m_j - m_i
    along = -(d @ u)                               # >0: j sits upwind of i
    off = ~np.eye(len(m), dtype=bool)
    strong = off & (peak > 0.3)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    ax = axes[0]
    ax.scatter(along[strong], lag_min[strong], s=12, color=SERIES[0], alpha=0.6,
               linewidths=0, label="link pairs (peak corr > 0.3)")
    v = cfg["storm_speed_km_min"]
    xs = np.linspace(along[strong].min() if strong.any() else -10,
                     along[strong].max() if strong.any() else 10, 50)
    ax.plot(xs, xs / v, color=INK_2, lw=1.2, ls="--", label=f"advection, v = {v:g} km/min")
    H_min = cfg.get("horizon_min", 5.0)
    ax.axhspan(0, H_min, color=SERIES[2], alpha=0.12, linewidth=0,
               label=f"useful lead: 0–{H_min:g} min horizon")
    ax.axhline(0, color=GRID, lw=1)
    ax.legend(loc="upper left")
    _ax(ax, "along-storm separation, source upwind of target [km]",
        "lead of source over target [min]", "(a) temporal scale: lead ∝ distance / speed",
        grid="both")

    ax = axes[1]
    dist = np.linalg.norm(d, axis=-1)
    ax.scatter(dist[off], peak[off], s=10, color=SERIES[0], alpha=0.45, linewidths=0,
               label="link pairs")
    # Fit the decay scale to the pairs shown rather than drawing the cells'
    # nominal sigma: path averaging and advection along the chain keep link
    # correlation higher than a single static cell of mean size would.
    ok = off & np.isfinite(peak)
    grid = np.linspace(0.5, 4 * dist[ok].max(), 400)
    sse = [np.sum((peak[ok] - np.exp(-dist[ok] ** 2 / (4 * g ** 2))) ** 2) for g in grid]
    sig = float(grid[int(np.argmin(sse))])
    dd = np.linspace(0, dist[off].max(), 100)
    ax.plot(dd, np.exp(-dd ** 2 / (4 * sig ** 2)), color=INK_2, lw=1.2, ls="--",
            label=f"Gaussian fit, σ = {sig:.1f} km")
    ax.set_ylim(min(0, np.nanmin(peak[off])), 1.02)
    ax.legend(loc="upper right")
    _ax(ax, "link separation [km]", "peak lagged correlation",
        "(b) spatial scale: coupling decays with distance", grid="both")
    fig.tight_layout()
    return _finish(fig, path)


# ---- attention --------------------------------------------------------------
def plot_spatial_attention(links, lam, target=None, storm_dir_deg=None, extent_km=None,
                           top_k=5, path=None, title=None):
    """Learned spatial attention drawn on the map.

    lam: (N, N) time-averaged Lambda, row i = distribution over sources j for
    target i. Arrows run source → target, width and opacity ∝ Lambda[i, j].
    With `target=None` the strongest `top_k` edges per target are drawn for
    every link; otherwise only the chosen target's sources.
    """
    style()
    lam = np.asarray(lam)
    N = lam.shape[0]
    m = _mids(links)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    _draw_links(ax, links, label_ids=True)
    targets = range(N) if target is None else [target]
    edges = []
    for i in targets:
        row = lam[i].copy()
        row[i] = 0.0
        for j in np.argsort(-row)[:top_k]:
            if row[j] > 0:
                edges.append((i, j, row[j]))
    wmax = max((w for *_, w in edges), default=1.0)
    for i, j, w in sorted(edges, key=lambda e: e[2]):
        a = w / wmax
        ax.annotate("", xy=m[i], xytext=m[j],
                    arrowprops=dict(arrowstyle="-|>", color=SERIES[0], lw=0.4 + 2.6 * a,
                                    alpha=0.25 + 0.75 * a, shrinkA=3, shrinkB=3,
                                    connectionstyle="arc3,rad=0.12"))
    if target is not None:
        ax.scatter(*m[target], s=90, facecolors="none", edgecolors=SERIES[1], lw=2, zorder=4)
    if extent_km is None:
        pts = np.array([p for l in links for p in (l["a"], l["b"])])
        extent_km = float(pts.max())
    if storm_dir_deg is not None:
        _storm_arrow(ax, extent_km, storm_dir_deg)
    ax.set_xlim(-0.02 * extent_km, 1.02 * extent_km)
    ax.set_ylim(-0.02 * extent_km, 1.02 * extent_km)
    ttl = title or ("Spatial attention Λ: source → target" if target is None
                    else f"Sources selected for link {target}")
    _ax(ax, "x [km]", "y [km]", ttl, grid=None)
    return _finish(fig, path)


def plot_attention_matrices(lam_learned, lam_star=None, path=None):
    """Learned Lambda next to the simulator's ground-truth influence Lambda*."""
    style()
    mats = [(lam_learned, "learned Λ")]
    if lam_star is not None:
        mats.insert(0, (lam_star, "ground truth Λ* (upwind, in-horizon)"))
    fig, axes = plt.subplots(1, len(mats), figsize=(4.1 * len(mats), 3.6), squeeze=False)
    for ax, (mat, t) in zip(axes[0], mats):
        # Row-normalise to each row's max: the comparison of interest is which
        # sources each target prefers, not the absolute level.
        mm = mat / np.maximum(mat.max(1, keepdims=True), 1e-12)
        im = ax.imshow(mm, cmap=SEQ, vmin=0, vmax=1)
        _ax(ax, "source link j", "target link i", t, grid=None)
    _colorbar(fig, im, axes[0][-1], "row-normalised weight")
    fig.tight_layout()
    return _finish(fig, path)


def plot_temporal_attention(pi_fwd, pi_bwd=None, dt_min=0.5, path=None):
    """Bi-temporal attention: which past steps each forecast step reads.

    pi: (H, T). Columns are past steps (right = most recent), rows are forecast
    steps. Forward and backward passes are shown separately because S-BSTN keeps
    them separate until the context vectors are fused.
    """
    style()
    mats = [(pi_fwd, "(a) forward pass")]
    if pi_bwd is not None:
        mats.append((pi_bwd, "(b) backward pass"))
    H, T = np.asarray(pi_fwd).shape
    fig, axes = plt.subplots(1, len(mats), figsize=(4.6 * len(mats), 3.2), squeeze=False)
    ext = [-(T - 0.5) * dt_min, 0.5 * dt_min, H * dt_min + 0.5 * dt_min, 0.5 * dt_min]
    vmax = max(float(np.max(p)) for p, _ in mats)
    for ax, (p, t) in zip(axes[0], mats):
        im = ax.imshow(p, cmap=SEQ, aspect="auto", vmin=0, vmax=vmax, extent=ext)
        _ax(ax, "past step [min relative to now]", "forecast lead [min]", t, grid=None)
    cb = _colorbar(fig, im, axes[0][-1], "attention π")
    # A flat panel at this level means the pass weights every past step equally.
    cb.ax.axhline(1.0 / T, color=INK, lw=1.5)
    cb.set_ticks([0, 1.0 / T, vmax], labels=["0", "1/T (uniform)", f"{vmax:.2f}"])
    fig.tight_layout()
    return _finish(fig, path)


# ---- forecasts & skill ------------------------------------------------------
def plot_forecast(x_hist, y_true, y_pred, links_idx=None, dt_min=0.5, path=None,
                  title="Forecast vs observed"):
    """One window: past T steps, then the H-step forecast against truth.

    x_hist: (N, T), y_true / y_pred: (N, H), all dB.
    """
    style()
    x_hist, y_true, y_pred = map(np.asarray, (x_hist, y_true, y_pred))
    N, T = x_hist.shape
    H = y_true.shape[1]
    if links_idx is None:
        links_idx = np.argsort(-(y_true.max(1) - x_hist.min(1)))[:4]
    tp = (np.arange(T) - T + 1) * dt_min
    tf = np.arange(1, H + 1) * dt_min
    n = len(links_idx)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 2.8), sharex=True, squeeze=False)
    for ax, i in zip(axes[0], links_idx):
        ax.plot(tp, x_hist[i], color=INK_2, lw=1.4)
        ax.plot(np.r_[0, tf], np.r_[x_hist[i, -1], y_true[i]], color=INK_2, lw=1.4,
                ls=":")
        ax.plot(np.r_[0, tf], np.r_[x_hist[i, -1], y_pred[i]], color=SERIES[0], lw=2)
        ax.axvline(0, color=GRID, lw=1)
        _ax(ax, "time [min]", "attenuation [dB]" if ax is axes[0][0] else None,
            f"link {i}")
    handles = [Line2D([], [], color=INK_2, lw=1.4, label="observed"),
               Line2D([], [], color=INK_2, lw=1.4, ls=":", label="future (truth)"),
               Line2D([], [], color=SERIES[0], lw=2, label="S-BSTN forecast")]
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(title, x=0.01, ha="left", y=1.13, fontsize=10, fontweight="semibold")
    fig.tight_layout()
    return _finish(fig, path)


def plot_horizon(results, path=None, key="rmse_avg", split="wet", dt_min=0.5,
                 arms=None):
    """Per-horizon error for each trained arm (results = run_scenario output)."""
    style()
    arms = arms or [a for a in ("persistence", "LSTM-ED", "STANN", "S-BSTN")
                    if a in results["arms"]] or list(results["arms"])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for a in arms:
        m = results["arms"][a]["metrics"][split]
        keys = sorted((k for k in m if k.endswith("min")), key=lambda k: float(k[:-3]))
        xs = [float(k[:-3]) for k in keys]
        ys = [m[k][key] for k in keys]
        ax.plot(xs, ys, marker="o", color=ARM_COLOR.get(a, INK), label=a)
    ax.legend(ncol=2, loc="upper left")
    _ax(ax, "forecast horizon [min]", f"{split}-period {key.replace('_', ' ').upper()} [dB]",
        "Forecast error by horizon (lower is better)")
    return _finish(fig, path)


def plot_ablation(results, path=None, key="rmse_avg", split="wet"):
    """Window-averaged error (TNERROR) per variant, sorted, full model highlighted."""
    style()
    arms = list(results["arms"])
    vals = [results["arms"][a]["metrics"][split]["TN"][key] for a in arms]
    idx = np.argsort(vals)
    arms = [arms[i] for i in idx]
    vals = [vals[i] for i in idx]
    colors = [SERIES[0] if a == "S-BSTN" else INK_MUTED if a == "persistence"
              else "#c9d6e8" for a in arms]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(arms) + 1.0))
    b = ax.barh([ABLATION_LABEL.get(a, a) for a in arms], vals, color=colors,
                linewidth=0, height=0.66)
    for rect, v in zip(b, vals):
        ax.text(rect.get_width() * 1.01, rect.get_y() + rect.get_height() / 2,
                f"{v:.3f}", va="center", color=INK_2, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(right=max(vals) * 1.13)
    _ax(ax, f"{split}-period TNERROR {key.replace('_', ' ').upper()} [dB]", None,
        "Ablation (lower is better)", grid="x")
    return _finish(fig, path)


def plot_training(history, path=None):
    """Loss curves, the learned SAN threshold tau, and the empty-row rate."""
    style()
    has_tau = bool(history.get("tau"))
    fig, axes = plt.subplots(1, 3 if has_tau else 1, figsize=(11 if has_tau else 4.2, 3.2),
                             squeeze=False)
    ax = axes[0][0]
    ep = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(ep, history["train_loss"], color=SERIES[0], label="train")
    ax.plot(ep, history["val_loss"], color=SERIES[1], label="validation")
    ax.set_yscale("log")
    ax.legend()
    _ax(ax, "epoch", "MSE (scaled units)", "(a) loss")
    if has_tau:
        tau = np.array(history["tau"])
        ax = axes[0][1]
        ax.plot(ep, tau.mean(1), color=SERIES[0], label="mean τ")
        ax.fill_between(ep, tau.min(1), tau.max(1), color=SERIES[0], alpha=0.16, lw=0,
                        label="per-link range")
        n = tau.shape[1]
        ax.axhline(1.0 / n, color=INK_MUTED, lw=1, ls="--")
        ax.annotate(f"uniform attention 1/N = {1 / n:.3f}", (ep[-1], 1.0 / n),
                    xytext=(-4, 4), textcoords="offset points", ha="right",
                    color=INK_2, fontsize=8)
        ax.legend(loc="lower left")
        _ax(ax, "epoch", "selection threshold τ", "(b) SAN threshold")
        ax = axes[0][2]
        ax.plot(ep, 100 * np.array(history["empty_row_rate"]), color=SERIES[3])
        _ax(ax, "epoch", "rows pruned empty [%]", "(c) SAN empty-row rate")
    fig.tight_layout()
    return _finish(fig, path)


# ---- downstream -------------------------------------------------------------
def plot_rain_map(scores, path=None, sample=0, horizon=-1):
    """Rain map from observed vs forecast attenuation, and their difference.

    `scores` is the output of sbstn.evaluate.rain_map_scores.
    """
    from .evaluate import idw_map
    style()
    g, mids = scores["grid"], scores["mids"]
    mt = idw_map(scores["r_true"][sample, :, horizon], mids, g, g)
    mp = idw_map(scores["r_pred"][sample, :, horizon], mids, g, g)
    vmax = max(mt.max(), mp.max(), 1e-6)
    ext = [g[0], g[-1], g[0], g[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, mm, t in ((axes[0], mt, "(a) from observed attenuation"),
                      (axes[1], mp, "(b) from S-BSTN forecast")):
        im = ax.imshow(mm, cmap=SEQ, origin="lower", vmin=0, vmax=vmax, extent=ext)
        ax.scatter(mids[:, 0], mids[:, 1], s=6, color=INK, linewidths=0)
        _ax(ax, "x [km]", "y [km]", t, grid=None)
    _colorbar(fig, im, axes[1], "rain rate [mm/h]")
    dd = mp - mt
    lim = max(abs(dd).max(), 1e-6)
    im2 = axes[2].imshow(dd, cmap=DIV, origin="lower", vmin=-lim, vmax=lim, extent=ext)
    _ax(axes[2], "x [km]", None, "(c) forecast − observed", grid=None)
    _colorbar(fig, im2, axes[2], "mm/h")
    if scores.get("placeholder_power_law"):
        fig.text(0.01, -0.03, "Placeholder power law (k=0.1, α=1): rain rates are "
                 "illustrative, not physically calibrated.", fontsize=7, color=INK_MUTED)
    fig.tight_layout()
    return _finish(fig, path)
