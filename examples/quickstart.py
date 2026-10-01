"""End-to-end S-BSTN in Python: simulate -> train -> forecast -> inspect attention.

    python examples/quickstart.py            # ~7 min on a laptop CPU
"""
from pathlib import Path

import numpy as np

from sbstn import Forecaster, viz
from sbstn.config import load
from sbstn.data.simulate import simulate
from sbstn.train import build_dataset, model_kwargs, run

OUT = Path("runs/example")

# 1. A 16-link chain over 45 km crossed by storms moving east at 30 km/h: the
#    regime S-BSTN is built for (see configs/quickstart.yaml).
cfg = load("configs/quickstart.yaml")
sim = simulate(cfg.sim)
print(f"simulated {sim['x_obs'].shape[0]} links, wet {sim['wet_mask'].mean():.1%} of the time")

# 2. Wet-dry balancing + sliding windows: X (S, N, T) -> Y (S, N, H), in dB.
X, Y, starts, wet, diag = build_dataset(sim, cfg.data)
print(f"{len(X)} windows after balancing, {diag['wet_window_frac']:.0%} of them wet")

# 3. Train S-BSTN (chronological split, early stopping) and keep it as a Forecaster.
result, model = run("S-BSTN", X, Y, wet, starts, cfg=cfg)
fc = Forecaster(model=model, scaler=model.scaler, dt_min=0.5,
                model_kwargs=model_kwargs(cfg.model), links=sim["links"])
fc.save(OUT / "sbstn.pt")
print("wet-period RMSE by horizon [dB]:",
      {k: round(v["rmse_avg"], 3) for k, v in result["metrics"]["wet"].items()})

# 4. Forecast the next 5 minutes for every link from the last 12 minutes.
test = result["_arrays"]
x_now = test["x_db"][-1]                         # (N, T) dB
y_hat = Forecaster.load(OUT / "sbstn.pt").predict(x_now)
print("forecast shape:", y_hat.shape, "(links x 30 s steps)")

# 5. What did it attend to? Spatial Lambda on the map, temporal pi over the window.
attn = fc.attention(test["x_db"][test["wet"]])
lam = attn["lambda_fwd"].mean(0)
viz.plot_spatial_attention(sim["links"], lam, storm_dir_deg=cfg.sim.storm_dir_deg,
                           extent_km=cfg.sim.extent_km, top_k=2,
                           path=OUT / "spatial_attention.png")
viz.plot_temporal_attention(attn["pi_fwd"], attn["pi_bwd"], fc.dt_min,
                            path=OUT / "temporal_attention.png")
viz.plot_spatiotemporal_structure(sim, path=OUT / "spatiotemporal_structure.png")
w_idx = np.flatnonzero(test["wet"])                # strongest wet event in the test set
swing = np.concatenate([test["x_db"], test["y_db"]], -1)
k = int(w_idx[np.argmax((swing.max(-1) - swing.min(-1)).max(-1)[w_idx])])
viz.plot_forecast(test["x_db"][k], test["y_db"][k], test["pred_db"][k],
                  path=OUT / "forecast.png")
print(f"figures in {OUT}/")
