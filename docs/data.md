# Data

## Using your own CML network

S-BSTN needs only the links' own attenuation records. Put them in an `.npz` file:

| Array | Shape | Required | Meaning |
|---|---|:-:|---|
| `x_obs` | `(N, L)` float | ✓ | attenuation `TSL − RSL` in dB, one row per link, evenly sampled |
| `dt_s` | scalar | | raw sampling interval in seconds (default 10) |
| `links` | JSON string | | `[{"id": 0, "a": [x, y], "b": [x, y], "freq_ghz": 23.0}, …]`, coordinates in km, used for maps and rain-rate conversion |

```python
import json, numpy as np
np.savez("my_network.npz", x_obs=x, dt_s=10.0, links=json.dumps(links))
```

```bash
sbstn train --data my_network.npz --out runs/mynet
```

Preprocessing follows the paper and uses only network data. Rain gauges and radar are never
used as inputs.

1. **Causal downsampling**, `data.downsample` (10 s → 30 s by default). This is a
   backward-looking moving average, because a centred window would leak the future.
2. **Wet/dry classification** using the rolling standard deviation over `data.window`
   samples, thresholded at the `1 − s` quantile. The threshold is fitted on the training
   portion only.
3. **Wet–dry balancing.** If *any* link is wet over `[t₁, t₂]`, the interval
   `[t₁ − pre, t₂ + post]` is kept for *all* links. The post margin captures wet-antenna
   decay. This raises the wet fraction from about 2% in raw records to about 35–40%.
4. **Sliding windows** of `T` past steps and `H` future steps.
5. **Chronological split** into 75/15/10 train/validation/test, with an embargo of `T + H`
   steps at each boundary so that overlapping windows cannot leak across the split.
6. **Per-link min-max scaling**, fitted on training data only. All metrics are reported in
   dB, after inverting the scaling.

Missing samples should be interpolated or masked before training. Links whose sampling or
baseline changes abruptly (for example after equipment replacement) should be split or
removed.

The Ericsson dataset used in the paper is proprietary and is not distributed here. A public
subset of the same Gothenburg network is available as
[OpenMRG](https://doi.org/10.5194/essd-14-5411-2022) (Andersson et al., 2022).

## The synthetic CML network simulator

[`sbstn/data/simulate.py`](../sbstn/data/simulate.py) generates CML records for which the
true spatial coupling is **known**, because the storm velocity is a parameter you choose.
You can use it to check that a pipeline is correct, to study attention maps against ground
truth, and to try out S-BSTN without access to operator data.

```
rain field R(x, y, t)                  advecting Gaussian cells (or VAR / frontal / multiscale / seasonal)
  → path-average over each link        r^{(i)}_t            [mm/h]
  → power law  γ = k r^α               specific attenuation [dB/km]
  → × path length                      A^{(i)}_t            [dB]
  → + baseline, drift, wet antenna, noise
  → quantise 0.3 dB, sample 10 s       x^{(i)}_t            (observed)
```

| Setting | Options |
|---|---|
| `sim.topology` | `chain`, `ring`, `tree`, `random`: similar to the paper's D3, D2 and D1 subnetworks, plus random |
| `sim.field_mode` | `advect` (default), `static`, `uniform`, `independent` (no shared structure, used as a control), `var`, `frontal`, `multiscale`, `seasonal` |
| `sim.n_links`, `sim.extent_km` | network size and spatial spread |
| `sim.storm_speed_km_min`, `sim.storm_dir_deg` | advection velocity, which sets the lead-lag structure |
| `sim.spawn_per_hour` | calibrated so that about 2% of the raw record is wet, matching the operational data |

```bash
sbstn simulate --config configs/base.yaml --set sim.topology=tree sim.n_links=34 --out data/tree34.npz
sbstn train --data data/tree34.npz --out runs/tree34
```

### ITU-R P.838-3 coefficients

By default the simulator and the rain-map module use a **placeholder** power law (`k = 0.1`,
`α = 1`) that does not depend on frequency. Forecasting in dB is unaffected, but rain rates in
mm/h are only illustrative. To use the real frequency- and polarisation-dependent law,
transcribe Tables 1–4 of Recommendation ITU-R P.838-3 into
`sbstn/data/itu_r_838_coeffs.json`. The expected format is documented in
[`itu_r_838.py`](../sbstn/data/itu_r_838.py). The tables are not bundled, so that the numbers
always come from the official recommendation.
