"""Inference API: a trained S-BSTN plus everything needed to use it.

A `Forecaster` bundles the network, the train-fitted scaler, the window
geometry and (optionally) the link metadata, and round-trips through a single
checkpoint file. Inputs and outputs are always in dB; scaling is internal.

    fc = Forecaster.load("runs/quickstart/sbstn.pt")
    y_hat = fc.predict(x_db)                 # (N, T) -> (N, H) dB
    attn = fc.attention(x_db)                # spatial Lambda, temporal pi
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data.scaling import MinMax
from .models.sbstn import SBSTN, build

CKPT_FORMAT = "sbstn-checkpoint/1"


@dataclass
class Forecaster:
    model: SBSTN
    scaler: MinMax
    variant: str = "S-BSTN"
    dt_min: float = 0.5
    model_kwargs: dict = field(default_factory=dict)
    links: list[dict] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def n_sensors(self) -> int:
        return self.model.N

    @property
    def t_window(self) -> int:
        return self.model.T

    @property
    def horizon(self) -> int:
        return self.model.H

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    # ------------------------------------------------------------------
    def _prep(self, x_db) -> tuple[torch.Tensor, bool]:
        x = np.asarray(x_db, dtype=np.float32)
        single = x.ndim == 2
        if single:
            x = x[None]
        if x.shape[1:] != (self.n_sensors, self.t_window):
            raise ValueError(
                f"expected (..., {self.n_sensors}, {self.t_window}) - "
                f"N links by T past steps at {self.dt_min:g} min - got {x.shape}"
            )
        return torch.as_tensor(self.scaler.transform(x), device=self.device), single

    @torch.no_grad()
    def predict(self, x_db, batch_size: int = 256) -> np.ndarray:
        """Forecast attenuation. (N, T) -> (N, H), or (B, N, T) -> (B, N, H), dB."""
        self.model.eval()
        x, single = self._prep(x_db)
        out = [self.model(x[k:k + batch_size]).cpu().numpy()
               for k in range(0, len(x), batch_size)]
        y = self.scaler.inverse(np.concatenate(out))
        return y[0] if single else y

    @torch.no_grad()
    def attention(self, x_db) -> dict[str, np.ndarray | None]:
        """Attention maps for one window or a batch (batch-averaged).

        lambda_fwd / lambda_bwd : (T, N, N)  spatial, row i = sources for target i
        pi_fwd / pi_bwd         : (H, T)     temporal, per forecast step
        """
        self.model.eval()
        x, _ = self._prep(x_db)
        self.model(x)
        a = self.model.last_attn

        def mean0(t):
            return None if t is None else t.mean(0).cpu().numpy()

        return dict(lambda_fwd=mean0(a["lambda_fwd"]), lambda_bwd=mean0(a["lambda_bwd"]),
                    pi_fwd=mean0(a["pi_fwd"]), pi_bwd=mean0(a["pi_bwd"]),
                    tau=self.tau())

    def tau(self) -> np.ndarray | None:
        """Learned per-link selection thresholds of the forward SCA block."""
        sca = getattr(getattr(self.model.encoder, "fwd", None), "sca", None)
        return None if sca is None else sca.tau.detach().cpu().numpy()

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(
            format=CKPT_FORMAT,
            variant=self.variant,
            shape=dict(n_sensors=self.n_sensors, t_window=self.t_window,
                       horizon=self.horizon),
            model_kwargs=self.model_kwargs,
            state_dict={k: v.cpu() for k, v in self.model.state_dict().items()},
            scaler=dict(lo=self.scaler.lo, hi=self.scaler.hi,
                        per_link=self.scaler.per_link),
            dt_min=self.dt_min,
            links=self.links,
            meta=self.meta,
        ), path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "Forecaster":
        ck = torch.load(path, map_location=device, weights_only=False)
        if ck.get("format") != CKPT_FORMAT:
            raise ValueError(f"{path}: not an S-BSTN checkpoint ({ck.get('format')!r})")
        shp = ck["shape"]
        model = build(ck["variant"], shp["n_sensors"], shp["t_window"], shp["horizon"],
                      **ck["model_kwargs"])
        model.load_state_dict(ck["state_dict"])
        model.to(device).eval()
        sc = MinMax(per_link=ck["scaler"]["per_link"])
        sc.lo, sc.hi = ck["scaler"]["lo"], ck["scaler"]["hi"]
        return cls(model=model, scaler=sc, variant=ck["variant"], dt_min=ck["dt_min"],
                   model_kwargs=ck["model_kwargs"], links=ck.get("links"),
                   meta=ck.get("meta", {}))
