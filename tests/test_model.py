"""Model invariants: shapes, attention distributions, gradient flow, causality."""
import pytest
import torch

from sbstn.models.sbstn import SBSTN, build, VARIANTS


def mk(N=7, T=24, H=10, **kw):
    return SBSTN(N, T, H, hidden=32, q_e=16, q_d=16, **kw)


@pytest.mark.parametrize("N,T", [(3, 3), (7, 24), (34, 24)])
def test_shapes(N, T):
    m = mk(N, T)
    out = m(torch.randn(4, N, T))
    assert out.shape == (4, N, 10)


def test_attention_rows_are_distributions():
    m = mk()
    m(torch.randn(4, 7, 24))
    lam = m.last_attn["lambda_fwd"]
    assert torch.allclose(lam.sum(-1), torch.ones_like(lam.sum(-1)), atol=1e-4)
    assert (lam >= 0).all()


def test_no_empty_rows_at_init():
    m = mk()
    m(torch.randn(8, 7, 24))
    assert m.last_attn["empty_row_rate"] == 0.0, "tau init is above uniform 1/N"


def test_tau_receives_gradient():
    """THE most important test here.

    The paper's hard threshold has zero gradient w.r.t. tau, so a literal
    implementation trains normally and leaves tau at its init forever. Only
    the straight-through estimator in sca.py makes this pass.
    """
    m = mk()
    out = m(torch.randn(4, 7, 24))
    out.sum().backward()
    g = m.encoder.fwd.sca.tau_raw.grad
    assert g is not None and g.abs().sum() > 0


def test_directions_are_independent():
    m = mk(bidirectional=True)
    assert m.encoder.fwd is not m.encoder.bwd
    a = m.encoder.fwd.sca.W_e.weight
    b = m.encoder.bwd.sca.W_e.weight
    assert a.data_ptr() != b.data_ptr()


def test_backward_states_in_real_time_order():
    """X zero except at t=0. The backward state at real time 0 must see it."""
    m = mk()
    X = torch.zeros(2, 7, 24)
    X[:, :, 0] = 5.0
    h_f, h_b, *_ = m.encoder(X)
    assert h_b[:, 0].abs().sum() > 0


def test_eval_path_never_reads_Y():
    """Y must not influence eval output, even with teacher forcing requested."""
    m = mk().eval()
    X = torch.randn(2, 7, 24)
    with torch.no_grad():
        base = m(X, Y=None)
        forced = m(X, Y=torch.full((2, 7, 10), 1e6), teacher_forcing_ratio=1.0)
    assert torch.equal(base, forced)


@pytest.mark.parametrize("v", sorted(VARIANTS))
def test_all_variants_instantiate(v):
    m = build(v, 7, 24, 10, hidden=32, q_e=16, q_d=16)
    assert m(torch.randn(2, 7, 24)).shape == (2, 7, 10)


def test_v4_has_fewer_params():
    full = sum(p.numel() for p in build("S-BSTN", 7, 24, 10, hidden=32).parameters())
    v4 = sum(p.numel() for p in build("S-BSTN-V4", 7, 24, 10, hidden=32).parameters())
    assert v4 < full


def test_overfits_single_batch():
    torch.manual_seed(0)
    m = mk()
    X, Y = torch.randn(4, 7, 24), torch.randn(4, 7, 10)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(300):
        opt.zero_grad()
        l = ((m(X) - Y) ** 2).mean()
        first = first if first is not None else l.item()
        l.backward(); opt.step()
    assert l.item() < 0.1 * first


def test_determinism():
    def once():
        torch.manual_seed(7)
        return mk()(torch.ones(2, 7, 24)).sum().item()
    assert once() == once()
