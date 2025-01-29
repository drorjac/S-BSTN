# Changelog

## 1.0.0 — 2025-01

First public release, accompanying the IEEE TIM paper.

- S-BSTN model: selective cross-attention (SCA) with learned SAN pruning,
  bidirectional encoder, bi-temporal attention, autoregressive LSTM decoder.
- Ablation variants V1–V4, STANN and LSTM-ED as presets of one class; persistence baseline.
- Synthetic CML network simulator with known ground-truth spatial coupling
  (advecting cells, VAR, frontal, multiscale and seasonal processes; chain, ring,
  tree and random topologies; wet antenna, drift, noise, 0.3 dB quantisation).
- Wet–dry balancing, causal downsampling, chronological splits with a leakage embargo.
- `Forecaster` inference API and checkpoint format; `sbstn` CLI
  (`simulate`, `train`, `predict`, `info`).
- Visualisation of spatial attention on the network map, temporal attention,
  spatio-temporal structure, forecasts, training dynamics and rain maps.
- Downstream attenuation → rain-rate → IDW rain-map evaluation.
