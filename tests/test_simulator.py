import numpy as np
from sbstn.data.simulate import SimConfig, simulate, empirical_influence
from sbstn.data import windowing as W


def small(**kw):
    return SimConfig(n_links=8, duration_min=60 * 24, **kw)


def test_generates_all_field_modes():
    for mode in ["advect", "uniform", "independent", "static"]:
        s = simulate(small(field_mode=mode))
        assert s["x_obs"].shape[0] == 8
        assert np.isfinite(s["x_obs"]).all()


def test_quantisation_and_units():
    s = simulate(small())
    q = s["config"]["quantization_dB"]
    r = s["x_obs"] / q
    assert np.allclose(r, np.round(r), atol=1e-3)


def test_reference_matches_empirical():
    """Geometry-derived Lambda* must agree with lagged cross-correlation.

    Disagreement means the field generator and the geometry disagree, which is
    a bug BEFORE any model is trained.
    """
    s = simulate(small(field_mode="advect", topology="chain"))
    emp = empirical_influence(s["r_true"], max_lag=30)
    ref = s["lambda_star"]
    assert np.corrcoef(emp.ravel(), ref.ravel())[0, 1] > 0.3


def test_independent_control_has_no_structure():
    s = simulate(small(field_mode="independent"))
    emp = empirical_influence(s["r_true"], max_lag=10)
    off = emp[~np.eye(8, dtype=bool)]
    assert off.std() < 0.15


def test_causal_downsample_does_not_leak():
    x = np.zeros((1, 30)); x[0, 20] = 1.0
    y = W.causal_downsample(x, 3)
    assert y[0, :6].sum() == 0


def test_lag_star_sign_matches_physical_lead():
    """lag*[i, j] > 0 must mean j is upwind and LEADS i. Checked against the
    lag that maximises the empirical cross-correlation of the rain field."""
    s = simulate(SimConfig(n_links=10, duration_min=60 * 24 * 5))
    r = s["r_true"].astype(float)
    r = r[:, r.max(0) > 0.1]
    z = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-9)
    i, j = 0, 3                                     # chain: j lies downwind of i
    assert s["lag_star"][i, j] < 0 < s["lag_star"][j, i]
    lags = range(-120, 121)
    c = [np.mean(z[j, max(L, 0): z.shape[1] + min(L, 0)] *
                 z[i, max(-L, 0): z.shape[1] - max(L, 0)]) for L in lags]
    best = list(lags)[int(np.argmax(c))]            # corr(r_j(t), r_i(t - L))
    assert best > 0, best                           # i leads j
    # Lambda* credits an adjacent upwind source (lead < horizon), never the
    # downwind one, and drops upwind sources whose lead exceeds the horizon.
    lam, lag = s["lambda_star"], s["lag_star"]
    assert lam[1, 0] > 0 and lam[0, 1] == 0
    assert lag[j, i] > 5.0 and lam[j, i] == 0
