"""Command-line interface.

    sbstn simulate  --config configs/quickstart.yaml --out data/sim.npz
    sbstn train     --config configs/quickstart.yaml --out runs/quickstart
    sbstn train     --data my_network.npz --out runs/mynet        # your own CMLs
    sbstn predict   --checkpoint runs/quickstart/sbstn.pt --input window.npy
    sbstn info      --checkpoint runs/quickstart/sbstn.pt

Any config value can be overridden inline: `--set sim.n_links=24 train.epochs=40`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__


# ---------------------------------------------------------------------------
def _parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def _config(args):
    from .config import load
    cfg = load(args.config) if args.config else load()
    over: dict[str, dict] = {}
    for item in args.set or []:
        key, _, val = item.partition("=")
        section, _, name = key.partition(".")
        if not name:
            raise SystemExit(f"--set expects section.key=value, got {item!r}")
        v = _parse_value(val)
        over.setdefault(section, {})[name] = tuple(v) if isinstance(v, list) else v
    return cfg.with_overrides(**over) if over else cfg


def _save_sim(sim: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_obs=sim["x_obs"], r_true=sim["r_true"], wet_mask=sim["wet_mask"],
        lambda_star=sim["lambda_star"], lag_star=sim["lag_star"],
        dt_s=np.float32(sim["config"]["dt_s"]),
        links=json.dumps(sim["links"]), config=json.dumps(sim["config"], default=list),
    )


def _load_npz(path: Path) -> dict:
    """Simulator output or user data. Required: x_obs (N, L) in dB.

    Optional: dt_s (default 10), links (JSON list of {id, a, b, freq_ghz}).
    """
    z = np.load(path, allow_pickle=False)
    if "x_obs" not in z:
        raise SystemExit(f"{path}: needs an array `x_obs` of shape (N, L) in dB")
    out = {k: z[k] for k in z.files}
    for k in ("links", "config"):
        if k in out:
            out[k] = json.loads(str(out[k]))
    return out


# ---------------------------------------------------------------------------
def cmd_simulate(args):
    from .data.simulate import simulate
    cfg = _config(args)
    sim = simulate(cfg.sim)
    out = Path(args.out)
    _save_sim(sim, out)
    N, L = sim["x_obs"].shape
    print(f"simulated {N} links x {L} samples "
          f"({L * cfg.sim.dt_s / 86400:.1f} days, wet {sim['wet_mask'].mean():.1%}) -> {out}")


def cmd_train(args):
    from . import viz
    from .data.simulate import simulate
    from .evaluate import rain_map_scores
    from .forecaster import Forecaster
    from .train import build_dataset, build_dataset_from_array, model_kwargs, run

    from .models.sbstn import VARIANTS
    if args.variant not in VARIANTS:
        raise SystemExit(f"--variant must be one of {sorted(VARIANTS)}, "
                         f"got {args.variant!r}")
    for b in args.baselines:
        if b not in VARIANTS and b != "persistence":
            raise SystemExit(f"--baselines accepts {sorted(VARIANTS) + ['persistence']}, "
                             f"got {b!r}")

    cfg = _config(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig = out / "figures"

    sim = None
    if args.data:
        d = _load_npz(Path(args.data))
        links = d.get("links")
        if "r_true" in d and "wet_mask" in d and d.get("config"):
            sim = dict(d, links=links)
            X, Y, starts, wet, diag = build_dataset(sim, cfg.data)
        else:
            X, Y, starts, wet, diag = build_dataset_from_array(d["x_obs"], cfg.data)
        dt_min = float(d.get("dt_s", 10.0)) * cfg.data.downsample / 60.0
    else:
        sim = simulate(cfg.sim)
        links = sim["links"]
        X, Y, starts, wet, diag = build_dataset(sim, cfg.data)
        dt_min = cfg.sim.dt_s * cfg.data.downsample / 60.0
    print(f"dataset: {len(X)} windows, N={X.shape[1]}, T={X.shape[2]}, H={Y.shape[2]}, "
          f"wet windows {diag['wet_window_frac']:.0%}")

    results = {"config": cfg.to_dict(), "data_diag": diag, "arms": {}}
    arms = [args.variant] + list(args.baselines)
    model_out = None
    for arm in dict.fromkeys(arms):
        r, model = run(arm, X, Y, wet, starts, cfg=cfg, dt_min=dt_min,
                       verbose=args.verbose)
        arrays = r.pop("_arrays")
        lam = r.pop("lambda_mean", None)
        pi = r.pop("pi_mean", None)
        r.pop("pi_bwd_mean", None)
        tau = r.pop("tau", None)
        if tau is not None:
            r["tau_final"] = tau.tolist()
        results["arms"][arm] = r
        tn = r["metrics"]["wet"].get("TN", {}).get("rmse_avg", float("nan"))
        print(f"  {arm:<12s} params={r['params']:>8,d}  wet RMSE={tn:.4f} dB  "
              f"({r['seconds']}s, {r['epochs_run']} ep)")
        if arm == args.variant:
            model_out = (model, r, arrays, lam, pi)

    model, r, arrays, lam, pi = model_out
    fc = Forecaster(model=model.cpu(), scaler=model.scaler, variant=args.variant,
                    dt_min=dt_min, model_kwargs=model_kwargs(cfg.model), links=links,
                    meta=dict(version=__version__, data_diag=diag,
                              test_metrics=r["metrics"]))
    ck = fc.save(out / "sbstn.pt")
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    np.savez_compressed(out / "test_predictions.npz", **arrays)
    print(f"checkpoint -> {ck}")

    if args.no_figures:
        return
    viz.plot_training(r["history"], fig / "training.png")
    k = int(np.argmax((arrays["y_db"].max(-1) - arrays["x_db"][..., -1]).max(-1)))
    viz.plot_forecast(arrays["x_db"][k], arrays["y_db"][k], arrays["pred_db"][k],
                      dt_min=dt_min, path=fig / "forecast.png")
    attn = fc.attention(arrays["x_db"][arrays["wet"]] if arrays["wet"].any()
                        else arrays["x_db"])
    if attn["pi_fwd"] is not None:
        viz.plot_temporal_attention(attn["pi_fwd"], attn["pi_bwd"], dt_min,
                                    fig / "temporal_attention.png")
    if attn["lambda_fwd"] is not None:
        lam_w = attn["lambda_fwd"].mean(0)
        storm = sim["config"]["storm_dir_deg"] if sim else None
        extent = sim["config"]["extent_km"] if sim else None
        viz.plot_attention_matrices(lam_w, sim["lambda_star"] if sim else None,
                                    fig / "spatial_attention_matrix.png")
        if links:
            viz.plot_spatial_attention(links, lam_w, storm_dir_deg=storm,
                                       extent_km=extent, top_k=2,
                                       path=fig / "spatial_attention_map.png")
    if links:
        viz.plot_network(links, sim["config"]["extent_km"] if sim else None,
                         sim["config"]["storm_dir_deg"] if sim else None,
                         path=fig / "network.png")
    if sim is not None:
        viz.plot_event(sim, fig / "event.png")
        viz.plot_spatiotemporal_structure(sim, fig / "spatiotemporal_structure.png")
        scores = rain_map_scores(arrays["y_db"], arrays["pred_db"], links, sim["config"],
                                 sim["config"]["extent_km"], n_grid=40)
        wet_idx = np.flatnonzero(arrays["wet"])
        sample = int(wet_idx[np.argmax(scores["r_true"][wet_idx, :, -1].sum(-1))]) \
            if wet_idx.size else 0
        viz.plot_rain_map(scores, fig / "rain_map.png", sample=sample)
    import matplotlib.pyplot as plt
    plt.close("all")
    print(f"figures    -> {fig}/")


def cmd_predict(args):
    from .forecaster import Forecaster
    fc = Forecaster.load(args.checkpoint, device=args.device)
    p = Path(args.input)
    x = np.load(p)
    if isinstance(x, np.lib.npyio.NpzFile):
        x = x[args.key]
    y = fc.predict(x)
    if args.out:
        np.save(args.out, y)
        print(f"forecast {y.shape} dB -> {args.out}")
    else:
        np.set_printoptions(precision=2, suppress=True, linewidth=120)
        print(y)


def cmd_info(args):
    import torch
    from .forecaster import Forecaster
    fc = Forecaster.load(args.checkpoint)
    n = sum(p.numel() for p in fc.model.parameters())
    tau = fc.tau()
    info = dict(variant=fc.variant, n_links=fc.n_sensors, t_window=fc.t_window,
                horizon=fc.horizon, dt_min=fc.dt_min,
                history_min=fc.t_window * fc.dt_min, lead_min=fc.horizon * fc.dt_min,
                parameters=n, model_kwargs=fc.model_kwargs,
                tau_mean=None if tau is None else float(tau.mean()),
                sbstn_version=fc.meta.get("version"), torch=torch.__version__)
    print(json.dumps(info, indent=2))


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="sbstn",
        description="S-BSTN: spatio-temporal attenuation forecasting for CML networks.")
    ap.add_argument("--version", action="version", version=f"sbstn {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_cfg(p):
        p.add_argument("--config", help="YAML config (default: configs/base.yaml)")
        p.add_argument("--set", nargs="*", metavar="SECTION.KEY=VALUE",
                       help="override config values, e.g. sim.n_links=24")

    p = sub.add_parser("simulate", help="generate a synthetic CML network record")
    add_cfg(p)
    p.add_argument("--out", default="data/sim.npz")
    p.set_defaults(fn=cmd_simulate)

    p = sub.add_parser("train", help="train, evaluate, checkpoint and plot")
    add_cfg(p)
    p.add_argument("--data", help=".npz with x_obs (N, L) dB; simulate if omitted")
    p.add_argument("--variant", default="S-BSTN")
    p.add_argument("--baselines", nargs="*", default=[],
                   help="optional extra variants to train alongside, e.g. persistence LSTM-ED")
    p.add_argument("--out", default="runs/latest")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("predict", help="forecast from a saved checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help=".npy (N, T) or (B, N, T) in dB, or .npz")
    p.add_argument("--key", default="x", help="array name inside an .npz input")
    p.add_argument("--out", help=".npy output path (prints if omitted)")
    p.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("info", help="describe a checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.set_defaults(fn=cmd_info)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
