"""Selective Cross-Attention (SCA) + Selective Attention Network (SAN).

Paper: SCA and SAN equations, Fig. 3 (assets/paper/fig3_sca_spatial.png).

Tested by tests/test_model.py and tests/test_pipeline.py - in particular
test_tau_receives_gradient and test_tau_actually_moves_under_training, which
together guard the straight-through estimator in _san.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SCA(nn.Module):
    """One direction's spatial cross-attention block.

    Scores every ordered pair (i, j) of sensors, softmaxes over sources j,
    optionally prunes with a per-target learnable threshold, and emits the
    spatial feature tensor.

    Shapes
    ------
    P, Q : (B, N, Q_e)   loop-invariant projections of X, see `project`
    h, c : (B, M)        previous encoder LSTM state
    x_t  : (B, N)        current reading of all sensors
    ->
    Lambda : (B, N, N)   rows sum to 1 over dim 2 (sources j)
    feat   : (B, N) if reduce == "rowsum" else (B, N * N)
    """

    def __init__(
        self,
        n_sensors: int,
        t_window: int,
        hidden: int,
        q_e: int,
        *,
        use_san: bool = True,
        learn_tau: bool = True,
        tau_init_frac: float = 0.5,
        tau_temperature: float = 0.01,
        keep_self: bool = False,
        tau_scale: str = "absolute",   # absolute | uniform
        reduce: str = "rowsum",
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        assert reduce in ("rowsum", "flatten")
        assert tau_scale in ("absolute", "uniform")
        self.N = n_sensors
        self.T = t_window
        self.M = hidden
        self.Qe = q_e
        self.use_san = use_san
        self.keep_self = keep_self
        self.reduce = reduce
        self.chunk_size = chunk_size
        self.temperature = tau_temperature

        # eq. 5.1 parameters
        self.W_e = nn.Linear(2 * hidden, q_e, bias=False)   # (Q_e, 2M)
        self.U_s = nn.Linear(t_window, q_e, bias=False)     # (Q_e, T) target view
        self.V_s = nn.Linear(t_window, q_e, bias=False)     # (Q_e, T) source view
        self.b_e = nn.Parameter(torch.zeros(q_e))
        self.v_e = nn.Linear(q_e, 1, bias=False)            # (Q_e,)

        # SAN threshold, parameterised through a sigmoid so tau in (0, 1) by
        # construction. Never clamp - clamping zeroes the gradient at the
        # boundary, which is the same bug one level down.
        #
        # Init BELOW uniform attention 1/N so nothing is pruned at step 1.
        # Init ABOVE 1/N collapses every row immediately and the run is dead.
        # `tau_scale` decides what units the threshold lives in.
        #
        #   "absolute" (paper-faithful): tau = sigmoid(raw) in [0,1], compared
        #       directly against the softmax weights.
        #   "uniform":  tau = sigmoid(raw) * 2/N, i.e. measured in units of the
        #       uniform attention level 1/N.
        #
        # A softmax row over N sources has mean weight 1/N, so the level that
        # separates "strong" from "weak" sources scales as 1/N. "uniform"
        # expresses tau in those units, so the same learned value means the
        # same thing at every network size - useful when transferring settings
        # between small and large subnetworks. Watch `empty_row_rate` in the
        # training history: if it climbs above a few percent, tau is pruning
        # whole rows and "uniform" is the better choice.
        #
        # Both initialise to tau = tau_init_frac / N, so they start identically
        # and differ only in how they move.
        self.tau_scale = tau_scale
        if tau_scale == "uniform":
            self._tau_mult = 2.0 / n_sensors
            frac0 = tau_init_frac / 2.0              # -> tau0 = tau_init_frac/N
        else:
            self._tau_mult = 1.0
            frac0 = tau_init_frac / n_sensors
        raw0 = math.log(frac0 / (1.0 - frac0))
        self.tau_raw = nn.Parameter(
            torch.full((n_sensors,), raw0), requires_grad=learn_tau
        )

        # diagnostics, populated each forward
        self.last_empty_row_rate: float = 0.0
        self.last_sparsity: float = 1.0

    # ------------------------------------------------------------------
    @property
    def tau(self) -> torch.Tensor:
        return torch.sigmoid(self.tau_raw) * self._tau_mult      # (N,)

    def project(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Loop-invariant projections. X is (B, N, T).

        These depend only on X, so they are computed ONCE per forward pass.
        Recomputing them inside the time loop is the classic perf mistake here.
        """
        return self.U_s(X), self.V_s(X)                # (B, N, Q_e) each

    # ------------------------------------------------------------------
    def _scores(self, P, Q, h, c) -> torch.Tensor:
        """Pairwise scores e_t, shape (B, N, N). e[b, i, j] = source j -> target i."""
        R = self.W_e(torch.cat([h, c], dim=-1))        # (B, Q_e)

        if self.chunk_size is None:
            inner = torch.tanh(
                R[:, None, None, :] + P[:, :, None, :] + Q[:, None, :, :] + self.b_e
            )                                           # (B, N, N, Q_e)
            return self.v_e(inner).squeeze(-1)          # (B, N, N)

        # Chunk the target axis to bound the (B, N, N, Q_e) intermediate.
        # Peak activation ~ 2 * T * B * N^2 * Q_e * 4 bytes across both
        # directions and all time steps. At B=64, N=34, T=24, Q_e=64, fp32
        # that is roughly 913 MB for this one tensor.
        out = []
        for s in range(0, self.N, self.chunk_size):
            e = slice(s, min(s + self.chunk_size, self.N))
            inner = torch.tanh(
                R[:, None, None, :] + P[:, e, None, :] + Q[:, None, :, :] + self.b_e
            )
            out.append(self.v_e(inner).squeeze(-1))
        return torch.cat(out, dim=1)

    def _san(self, d: torch.Tensor) -> torch.Tensor:
        """Prune and renormalise. d is (B, N, N), already row-softmaxed."""
        tau = self.tau[None, :, None]                   # (1, N, 1) per TARGET i

        # Straight-through estimator. The paper's hard threshold has zero
        # gradient w.r.t. tau almost everywhere, so a literal implementation
        # trains fine and leaves tau at its initialisation FOREVER, silently.
        # Forward uses the hard mask, backward uses the sigmoid surrogate.
        soft = torch.sigmoid((d - tau) / self.temperature)
        hard = (d >= tau).to(d.dtype)
        mask = hard + soft - soft.detach()

        if self.keep_self:
            eye = torch.eye(self.N, device=d.device, dtype=d.dtype)
            mask = torch.clamp(mask + eye[None], max=1.0)

        d_tilde = d * mask

        # Empty-row guard: if tau_i exceeded every score in row i the row is
        # all zeros and renormalisation divides by zero. Fall back to the
        # single strongest source and LOG how often this fires - a rising
        # rate means tau is running away.
        row = d_tilde.sum(-1)                           # (B, N)
        empty = row < 1e-8
        if empty.any():
            top1 = F.one_hot(d.argmax(-1), self.N).to(d.dtype)
            d_tilde = torch.where(empty[..., None], d * top1, d_tilde)

        with torch.no_grad():
            self.last_empty_row_rate = empty.float().mean().item()
            self.last_sparsity = (d_tilde > 0).to(d.dtype).mean().item()

        return d_tilde / d_tilde.sum(-1, keepdim=True).clamp_min(1e-8)

    # ------------------------------------------------------------------
    def forward(self, P, Q, h, c, x_t):
        e = self._scores(P, Q, h, c)                    # (B, N, N)
        d = torch.softmax(e, dim=-1)                    # over sources j
        lam = self._san(d) if self.use_san else d

        # Entry (i, j) = Lambda[i, j] * x_t[j]  -- the SOURCE's reading.
        #
        # This is the matrix form Lambda_t @ diag(x_t). Scaling by the
        # TARGET's own x_t[i] instead could not be row-summed:
        # sum_j Lambda[i,j] * x[i] = x[i], the identity, so attention would
        # provably do nothing.
        feat_full = lam * x_t[:, None, :]                # (B, N, N)

        if self.reduce == "rowsum":
            feat = feat_full.sum(-1)                     # (B, N)
        else:
            feat = feat_full.reshape(feat_full.size(0), -1)   # (B, N*N)
        return lam, feat

    @property
    def out_dim(self) -> int:
        return self.N if self.reduce == "rowsum" else self.N * self.N
