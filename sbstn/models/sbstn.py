"""S-BSTN assembly. All ablation variants are config flags on this one class.

Paper: Fig. 2 (assets/paper/fig2_pipeline_bstn_x.png) end to end.
Architecture walkthrough: docs/architecture.md.
"""
from __future__ import annotations

import random
import torch
import torch.nn as nn

from .bita import BiTA, LastStateContext
from .decoder import DecoderInit, DecoderLSTM
from .encoder import BiEncoder


class SBSTN(nn.Module):
    def __init__(
        self,
        n_sensors: int,
        t_window: int,
        horizon: int,
        *,
        hidden: int = 64,
        q_e: int = 64,
        q_d: int = 64,
        bidirectional: bool = True,
        use_spatial_attention: bool = True,
        use_temporal_attention: bool = True,
        use_san: bool = True,
        learn_tau: bool = True,
        tau_init_frac: float = 0.5,
        tau_temperature: float = 0.01,
        keep_self: bool = False,
        tau_scale: str = "absolute",
        residual_head: bool = False,
        spatial_reduce: str = "rowsum",
        chunk_size: int | None = None,
    ):
        super().__init__()
        self.N, self.T, self.H, self.M = n_sensors, t_window, horizon, hidden

        self.encoder = BiEncoder(
            n_sensors, t_window, hidden, q_e,
            bidirectional=bidirectional,
            use_spatial_attention=use_spatial_attention,
            use_san=use_san,
            learn_tau=learn_tau,
            tau_init_frac=tau_init_frac,
            tau_temperature=tau_temperature,
            keep_self=keep_self,
            tau_scale=tau_scale,
            reduce=spatial_reduce,
            chunk_size=chunk_size,
        )

        self.context = (
            BiTA(hidden, q_d, bidirectional=bidirectional)
            if use_temporal_attention
            else LastStateContext(hidden, bidirectional=bidirectional)
        )

        # Decoder init: project the final encoder states rather than starting
        # from zeros.
        init_in = 2 * hidden if bidirectional else hidden
        self.dec_init = DecoderInit(init_in, hidden)
        self.decoder = DecoderLSTM(n_sensors, hidden, residual=residual_head)

    # ------------------------------------------------------------------
    def forward(self, X, Y=None, teacher_forcing_ratio: float = 0.0):
        """X: (B, N, T) -> Y_hat: (B, N, H).

        Y is only ever read when teacher_forcing_ratio > 0, which must be 0 at
        eval. tests/test_model.py asserts the eval path works with Y=None.
        """
        h_f, h_b, lam_f, lam_b, (hf, cf), (hb, cb) = self.encoder(X)

        if self.encoder.bidirectional:
            h0 = torch.cat([hf, hb], dim=-1)
            c0 = torch.cat([cf, cb], dim=-1)
        else:
            h0, c0 = hf, cf
        h_dec, c_dec = self.dec_init(h0, c0)

        y_prev = X[:, :, -1]            # last observed slice, NOT zeros
        outs, pis_f, pis_b = [], [], []

        for t in range(self.H):
            z, pi_f, pi_b = self.context(h_f, h_b, h_dec, c_dec)
            y_hat, h_dec, c_dec = self.decoder.step(y_prev, z, h_dec, c_dec)
            outs.append(y_hat)
            if pi_f is not None:
                pis_f.append(pi_f)
            if pi_b is not None:
                pis_b.append(pi_b)

            use_tf = (
                self.training
                and Y is not None
                and teacher_forcing_ratio > 0
                and random.random() < teacher_forcing_ratio
            )
            y_prev = Y[:, :, t] if use_tf else y_hat

        self.last_attn = {
            "lambda_fwd": lam_f,
            "lambda_bwd": lam_b,
            "pi_fwd": torch.stack(pis_f, 1) if pis_f else None,
            "pi_bwd": torch.stack(pis_b, 1) if pis_b else None,
            "empty_row_rate": self.encoder.fwd.last_empty_row_rate,
            "sparsity": self.encoder.fwd.last_sparsity,
        }
        return torch.stack(outs, dim=-1)                          # (B, N, H)


# ----------------------------------------------------------------------
# Ablation presets (paper, "Ablation Study").
VARIANTS = {
    "S-BSTN":     dict(),
    "S-BSTN-V1":  dict(use_spatial_attention=False),
    "S-BSTN-V2":  dict(use_temporal_attention=False),
    "S-BSTN-V3":  dict(use_san=False),
    "S-BSTN-V4":  dict(bidirectional=False),
    "STANN":      dict(bidirectional=False, use_san=False),
    "LSTM-ED":    dict(use_spatial_attention=False, use_temporal_attention=False,
                       use_san=False, bidirectional=False),
}


def build(variant: str, n_sensors, t_window, horizon, **base):
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}, have {sorted(VARIANTS)}")
    return SBSTN(n_sensors, t_window, horizon, **{**base, **VARIANTS[variant]})
