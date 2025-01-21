"""Experiment configuration.

One dataclass per concern, one `ExperimentConfig` holding them, loadable from
the YAML in `configs/`. Every run serialises its config and seed alongside its
results, so every number can be traced back to its settings.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any
import json

import yaml

from .data.simulate import SimConfig


@dataclass
class DataConfig:
    downsample: int = 3          # 10 s -> 30 s
    window: int = 60             # rolling-std window S, paper does not give this
    s: float = 0.012             # tuned: post-balancing wet fraction 0.360, paper band 33-44%
    pre: int = 20
    post: int = 40               # covers wet-antenna decay
    T: int = 24
    H: int = 10                  # 5 min at 30 s
    stride: int = 1              # window stride; >1 subsamples overlapping windows
    max_windows: int | None = 1200   # cap dataset size; equalises cost across scenarios
    embargo: int | None = None   # default T + H; 0 disables

    def resolved_embargo(self) -> int:
        return self.T + self.H if self.embargo is None else self.embargo


@dataclass
class ModelConfig:
    hidden: int = 32             # grid {16,32,64,128,256}; 64 makes the sweep infeasible
    q_e: int = 32
    q_d: int = 32
    spatial_reduce: str = "rowsum"   # rowsum | flatten
    tau_init_frac: float = 0.5       # -> tau0 = 0.5/N, BELOW uniform 1/N
    tau_temperature: float = 0.01
    learn_tau: bool = True
    keep_self: bool = False
    tau_scale: str = "absolute"   # absolute (paper) | uniform (threshold in units of 1/N)
    residual_head: bool = False   # DEVIATION: predict the change, not the level
    chunk_size: int | None = None


@dataclass
class TrainConfig:
    epochs: int = 150
    patience: int = 15
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 1e-5       # the paper's unspecified Omega(Theta)
    loss_reduction: str = "mean"     # `sum` is paper-faithful but scales with N
    teacher_forcing_ratio: float = 0.0
    optimizer: str = "adam"          # adam | sgd
    grad_clip: float = 5.0
    seed: int = 0
    device: str = "auto"


@dataclass
class ExperimentConfig:
    sim: SimConfig = field(default_factory=SimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return {k: asdict(v) for k, v in vars(self).items()}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))

    def with_overrides(self, **sections) -> "ExperimentConfig":
        """`cfg.with_overrides(sim=dict(n_links=40), model=dict(hidden=32))`."""
        out = {}
        for name, current in vars(self).items():
            over = sections.get(name)
            out[name] = replace(current, **over) if over else current
        return ExperimentConfig(**out)


def _tuple_fields(cls) -> set[str]:
    import typing
    hints = typing.get_type_hints(cls)
    return {k for k, v in hints.items() if getattr(v, "__origin__", None) is tuple}


def load(path: str | Path = None) -> ExperimentConfig:
    """Load `configs/base.yaml` (or another file) into an ExperimentConfig.

    YAML has no tuple type, so list-valued fields that the dataclasses declare
    as tuples are converted back.
    """
    if path is None:
        path = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
        if not path.exists():           # installed without the repo checkout
            return ExperimentConfig()
    raw = yaml.safe_load(Path(path).read_text()) or {}

    sections = {"sim": SimConfig, "data": DataConfig,
                "model": ModelConfig, "train": TrainConfig}
    built = {}
    for name, cls in sections.items():
        vals = dict(raw.get(name, {}) or {})
        for k in _tuple_fields(cls) & vals.keys():
            if isinstance(vals[k], list):
                vals[k] = tuple(vals[k])
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(vals) - known
        if unknown:
            raise KeyError(f"{path}: unknown key(s) in [{name}]: {sorted(unknown)}")
        built[name] = cls(**vals)
    return ExperimentConfig(**built)
