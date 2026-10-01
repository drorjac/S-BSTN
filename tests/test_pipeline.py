"""Pipeline tests: the scaling round-trip, gradients actually differing between
directions, leakage guards, the balancing pipeline landing in the paper's band,
and the persistence dry-period error sitting below quantisation.
"""
import numpy as np
import pytest
import torch

from sbstn.config import load
from sbstn.data import windowing as W
from sbstn.data.scaling import MinMax
from sbstn.data.simulate import simulate, empirical_influence
from sbstn.models.sbstn import build
from sbstn.train import build_dataset, run
from sbstn import metrics as M


# ---------------------------------------------------------------- scaling
def test_scaling_round_trip_and_train_only_fit():
    rng = np.random.default_rng(0)
    X = rng.normal(40, 5, (100, 6, 24)).astype("float32")
    tr = slice(0, 70)
    sc = MinMax().fit(X[tr])
    assert np.allclose(sc.inverse(sc.transform(X)), X, atol=1e-3)
    # Fit must depend only on the training slice.
    X2 = X.copy()
    X2[80:] += 1000.0
    sc2 = MinMax().fit(X2[tr])
    assert np.allclose(sc.lo, sc2.lo) and np.allclose(sc.hi, sc2.hi)


def test_metrics_are_in_dB_not_scaled_units():
    """The easiest way to produce numbers that look great and mean nothing."""
    rng = np.random.default_rng(0)
    y = rng.normal(40, 5, (50, 4, 10))
    sc = MinMax().fit(y.astype("float32"))
    scaled = sc.transform(y)
    m_scaled = M.report(scaled, scaled + 0.01)["TN"]["rmse_avg"]
    m_db = M.report(y, y + 0.01 * (sc.hi - sc.lo))["TN"]["rmse_avg"]
    assert m_db > 5 * m_scaled, "metrics appear to be on the [0,1] scale"


# ---------------------------------------------------------------- model
def test_direction_gradients_differ():
    """Directions are distinct - parameters AND gradients."""
    m = build("S-BSTN", 7, 24, 10, hidden=32, q_e=16, q_d=16)
    m(torch.randn(4, 7, 24)).sum().backward()
    gf = m.encoder.fwd.sca.W_e.weight.grad
    gb = m.encoder.bwd.sca.W_e.weight.grad
    assert gf is not None and gb is not None
    assert not torch.allclose(gf, gb)


def test_v4_halves_the_attention_parameters():
    """V4 has roughly half the ATTENTION parameters."""
    def attn_params(model):
        return sum(p.numel() for n, p in model.named_parameters()
                   if ".sca." in n or "context.fwd" in n or "context.bwd" in n)
    full = attn_params(build("S-BSTN", 12, 24, 10, hidden=32, q_e=16, q_d=16))
    v4 = attn_params(build("S-BSTN-V4", 12, 24, 10, hidden=32, q_e=16, q_d=16))
    assert 0.4 * full <= v4 <= 0.6 * full, (v4, full)


def test_tau_actually_moves_under_training():
    """The gradient test proves tau CAN move; this proves it DOES.

    A straight-through estimator that is wired up but scaled to nothing would
    still pass test_tau_receives_gradient.
    """
    torch.manual_seed(0)
    m = build("S-BSTN", 6, 12, 5, hidden=16, q_e=16, q_d=16)
    tau0 = m.encoder.fwd.sca.tau.detach().clone()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    X, Y = torch.randn(8, 6, 12), torch.randn(8, 6, 5)
    for _ in range(40):
        opt.zero_grad(); ((m(X) - Y) ** 2).mean().backward(); opt.step()
    assert (m.encoder.fwd.sca.tau.detach() - tau0).abs().max() > 1e-4


@pytest.mark.parametrize("reduce", ["rowsum", "flatten"])
def test_spatial_reduce_modes(reduce):
    m = build("S-BSTN", 5, 12, 4, hidden=16, q_e=16, q_d=16, spatial_reduce=reduce)
    assert m(torch.randn(3, 5, 12)).shape == (3, 5, 4)


def test_chunked_scores_match_unchunked():
    torch.manual_seed(0)
    a = build("S-BSTN", 8, 12, 4, hidden=16, q_e=16, q_d=16, chunk_size=None)
    b = build("S-BSTN", 8, 12, 4, hidden=16, q_e=16, q_d=16, chunk_size=3)
    b.load_state_dict(a.state_dict())
    X = torch.randn(2, 8, 12)
    a.eval(); b.eval()
    with torch.no_grad():
        assert torch.allclose(a(X), b(X), atol=1e-5)


# ---------------------------------------------------------------- data
def test_embargo_removes_boundary_overlap():
    starts = np.arange(2000)
    tr, va, te = W.chrono_split(2000, starts=starts, embargo=34)
    assert starts[tr].max() + 34 <= starts[va].min()
    assert starts[va].max() + 34 <= starts[te].min()


def test_embargo_raises_rather_than_silently_emptying_a_split():
    with pytest.raises(ValueError, match="embargo"):
        W.chrono_split(200, starts=np.arange(200), embargo=34)


def test_wet_dry_threshold_uses_training_portion_only():
    rng = np.random.default_rng(0)
    x = rng.normal(40, 0.1, (4, 2000))
    x_spiked = x.copy()
    x_spiked[:, 1900:] += 50.0          # huge excursion in the held-out tail
    _, s_a = W.wet_dry(x, fit_frac=0.75)
    _, s_b = W.wet_dry(x_spiked, fit_frac=0.75)
    assert abs(s_a - s_b) < 1e-9


# ---------------------------------------------------------------- gates
@pytest.mark.slow
def test_balancing_lands_in_paper_band():
    """Post-balancing wet fraction lands in the paper's 33-44% band."""
    cfg = load()
    sim = simulate(cfg.sim)
    *_, diag = build_dataset(sim, cfg.data)
    assert 0.018 <= diag["wet_frac_raw"] <= 0.023, diag["wet_frac_raw"]
    assert 0.33 <= diag["wet_frac_balanced"] <= 0.44, diag["wet_frac_balanced"]


@pytest.mark.slow
def test_persistence_dry_error_below_quantisation():
    """Reproduces the paper's observation that dry-period
    error sits below the 0.3 dB quantisation scale for every model, naive
    included. If this fails, the dry periods are not actually quiet."""
    cfg = load().with_overrides(sim=dict(n_links=8, duration_min=60 * 24 * 10))
    sim = simulate(cfg.sim)
    X, Y, starts, wet, _ = build_dataset(sim, cfg.data)
    r, _ = run("persistence", X, Y, wet, starts, cfg=cfg)
    dry = r["metrics"]["dry"]["TN"]["rmse_avg"]
    assert dry < cfg.sim.quantization_dB, dry


@pytest.mark.slow
def test_reference_matches_empirical_on_default_config():
    """Geometry-derived and correlation-derived Lambda* agree (r > 0.3).
    Without the horizon indicator in reference_influence this drops below."""
    cfg = load()
    s = simulate(cfg.sim)
    emp = empirical_influence(s["r_true"], max_lag=30)
    r = np.corrcoef(emp.ravel(), s["lambda_star"].ravel())[0, 1]
    assert r > 0.3, r


# ---------------------------------------------------------------- tau scaling
@pytest.mark.parametrize("N", [5, 16, 32])
def test_tau_scale_modes_share_initialisation(N):
    """Both parameterisations must start at tau = 0.5/N, so a comparison
    between them isolates how tau MOVES, not where it starts."""
    a = build("S-BSTN", N, 12, 5, hidden=16, q_e=16, q_d=16, tau_scale="absolute")
    b = build("S-BSTN", N, 12, 5, hidden=16, q_e=16, q_d=16, tau_scale="uniform")
    for m in (a, b):
        assert abs(float(m.encoder.fwd.sca.tau.mean()) - 0.5 / N) < 1e-4


def test_uniform_tau_scale_is_bounded_by_twice_uniform():
    """Under `uniform`, tau lives in units of 1/N and cannot exceed 2/N, which
    is what stops it drifting to a constant absolute value as N grows."""
    N = 32
    m = build("S-BSTN", N, 12, 5, hidden=16, q_e=16, q_d=16, tau_scale="uniform")
    with torch.no_grad():
        m.encoder.fwd.sca.tau_raw.fill_(20.0)        # saturate the sigmoid
    assert float(m.encoder.fwd.sca.tau.max()) <= 2.0 / N + 1e-6


def test_pick_device():
    from sbstn.train import pick_device
    assert pick_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        assert pick_device("auto", n_sensors=8).type == "cpu"     # small N stays on CPU
        if torch.backends.mps.is_available():
            assert pick_device("auto", n_sensors=64).type == "mps"
            assert pick_device("gpu").type == "mps"
