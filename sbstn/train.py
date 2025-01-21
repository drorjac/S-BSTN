"""Training loop + single-scenario experiment driver.

Every run emits: a per-horizon metric table in dB, the learned spatial (Lambda)
and temporal (pi) attention maps, the tau trajectory over training, the SAN
empty-row-rate curve, and the config + seed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import ExperimentConfig, DataConfig, ModelConfig
from .data.simulate import simulate
from .data import windowing as W
from .data.scaling import MinMax
from .models.sbstn import build
from .models.baselines import Persistence
from . import metrics as M


def pick_device(requested: str = "auto") -> torch.device:
    r = (requested or "auto").lower()
    if r in ("auto", "best", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        # MPS is available on this hardware but is SLOWER here: the encoder is
        # a Python loop over T steps with small per-step kernels, so launch
        # overhead dominates and CPU wins for these model sizes.
        return torch.device("cpu")
    return torch.device(r)


def loss_fn(yhat, y, reduction="mean"):
    """Paper sums over N and H. That makes the loss scale with N, so a LR tuned
    on N=7 will not transfer to N=34. Use mean while tuning, report which."""
    se = (yhat - y) ** 2
    return se.sum((1, 2)).mean() if reduction == "sum" else se.mean()


# ----------------------------------------------------------------------
def build_dataset(sim: dict, data: DataConfig):
    """Simulated record -> supervised pairs + pipeline diagnostics."""
    x = W.causal_downsample(sim["x_obs"], data.downsample)
    gt = W.causal_downsample(sim["wet_mask"].astype("float32"), data.downsample) > 0.5

    wet, sigma0 = W.wet_dry(x, data.window, data.s)
    keep = W.balanced_intervals(wet, data.pre, data.post)
    X, Y, starts, wet_flag = W.make_pairs(x, data.T, data.H, keep,
                                          wet_mask=gt, stride=data.stride)

    # Cap the dataset uniformly in time. Without this, scenarios differ wildly
    # in size for reasons unrelated to what is being measured: the balancing
    # rule keeps an interval if ANY link is wet, so with N=16 independent links
    # each raining ~3% of the time, "any link wet" holds ~39% of the time and
    # the `independent` control yields 9,331 windows against advect's 1,171 -
    # eight times the data and eight times the training cost. Capping removes
    # that confound and bounds the sweep budget.
    n_uncapped = len(X)
    if data.max_windows and n_uncapped > data.max_windows:
        sel = np.linspace(0, n_uncapped - 1, data.max_windows).round().astype(int)
        sel = np.unique(sel)
        X, Y, starts, wet_flag = X[sel], Y[sel], starts[sel], wet_flag[sel]

    # Validate the classifier against ground truth. You cannot do this on real
    # data - it is the whole reason for simulating first.
    tp = int((wet & gt).sum()); fp = int((wet & ~gt).sum()); fn = int((~wet & gt).sum())
    diag = dict(
        precision=float(tp / max(tp + fp, 1)),
        recall=float(tp / max(tp + fn, 1)),
        wet_frac_raw=float(gt.mean()),
        wet_frac_balanced=float(gt[:, keep].mean()) if keep.any() else 0.0,
        kept_frac=float(keep.mean()),
        sigma0=float(sigma0),
        n_windows=int(len(X)),
        n_windows_uncapped=int(n_uncapped),
        wet_window_frac=float(wet_flag.mean()),
    )
    return X, Y, starts, wet_flag, diag


def model_kwargs(mcfg: ModelConfig) -> dict:
    """ModelConfig -> SBSTN constructor kwargs. Stored in every checkpoint."""
    return dict(hidden=mcfg.hidden, q_e=mcfg.q_e, q_d=mcfg.q_d,
                spatial_reduce=mcfg.spatial_reduce, tau_init_frac=mcfg.tau_init_frac,
                tau_temperature=mcfg.tau_temperature, learn_tau=mcfg.learn_tau,
                keep_self=mcfg.keep_self, tau_scale=mcfg.tau_scale,
                residual_head=mcfg.residual_head,
                chunk_size=mcfg.chunk_size)


def build_dataset_from_array(x_obs: np.ndarray, data: DataConfig):
    """Real (or any unlabelled) record -> supervised pairs + diagnostics.

    x_obs: (N, L) attenuation in dB at the raw sampling interval. With no rain
    ground truth available, the rolling-std detector's own wet flags drive both
    the balancing and the wet/dry metric split.
    """
    x = W.causal_downsample(np.asarray(x_obs, dtype="float32"), data.downsample)
    wet, sigma0 = W.wet_dry(x, data.window, data.s)
    keep = W.balanced_intervals(wet, data.pre, data.post)
    X, Y, starts, wet_flag = W.make_pairs(x, data.T, data.H, keep,
                                          wet_mask=wet, stride=data.stride)
    n_uncapped = len(X)
    if data.max_windows and n_uncapped > data.max_windows:
        sel = np.unique(np.linspace(0, n_uncapped - 1, data.max_windows).round().astype(int))
        X, Y, starts, wet_flag = X[sel], Y[sel], starts[sel], wet_flag[sel]
    diag = dict(
        wet_frac_detected=float(wet.mean()),
        wet_frac_balanced=float(wet[:, keep].mean()) if keep.any() else 0.0,
        kept_frac=float(keep.mean()),
        sigma0=float(sigma0),
        n_windows=int(len(X)),
        n_windows_uncapped=int(n_uncapped),
        wet_window_frac=float(wet_flag.mean()),
    )
    return X, Y, starts, wet_flag, diag


def _make_model(variant, N, T, H, mcfg: ModelConfig):
    if variant == "persistence":
        return Persistence(H)
    return build(variant, N, T, H, **model_kwargs(mcfg))


def _sca_modules(model):
    enc = getattr(model, "encoder", None)
    out = []
    if enc is None:
        return out
    for d in (getattr(enc, "fwd", None), getattr(enc, "bwd", None)):
        if d is not None and hasattr(d, "sca"):
            out.append(d.sca)
    return out


def run(variant: str, X, Y, wet_flag, starts, *, cfg: ExperimentConfig,
        verbose=False):
    """Train one arm and evaluate it. Returns a result dict."""
    tc, mc, dc = cfg.train, cfg.model, cfg.data
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)

    S, N, T = X.shape
    H = Y.shape[-1]
    device = pick_device(tc.device)
    # Embargo on real time, not window index: consecutive windows overlap by
    # T+H-1 steps, so adjacent windows either side of a split boundary share
    # nearly all their samples.
    tr, va, te = W.chrono_split(S, starts=starts, embargo=dc.resolved_embargo())

    # Scaler fit on TRAIN ONLY.
    sc = MinMax().fit(X[tr])
    Xs, Ys = sc.transform(X), sc.transform(Y)
    t = lambda a: torch.as_tensor(a, device=device)

    model = _make_model(variant, N, T, H, mc).to(device)
    n_par = sum(p.numel() for p in model.parameters())

    history = {"train_loss": [], "val_loss": [], "tau_mean": [],
               "tau": [], "empty_row_rate": [], "sparsity": []}
    t_start = time.time()

    if variant != "persistence":
        opt = (torch.optim.Adam(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
               if tc.optimizer == "adam" else
               torch.optim.SGD(model.parameters(), lr=tc.lr, momentum=0.9,
                               weight_decay=tc.weight_decay))
        best, bad, best_state = float("inf"), 0, None
        scas = _sca_modules(model)

        for ep in range(tc.epochs):
            model.train()
            perm = np.random.permutation(tr)
            tot, nb = 0.0, 0
            for k in range(0, len(perm), tc.batch_size):
                idx = perm[k:k + tc.batch_size]
                opt.zero_grad()
                out = model(t(Xs[idx]), t(Ys[idx]),
                            teacher_forcing_ratio=tc.teacher_forcing_ratio)
                l = loss_fn(out, t(Ys[idx]), tc.loss_reduction)
                l.backward()
                nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                opt.step()
                tot += float(l.item()); nb += 1

            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(t(Xs[va])), t(Ys[va]), tc.loss_reduction).item()

            history["train_loss"].append(tot / max(nb, 1))
            history["val_loss"].append(vl)
            if scas:
                taus = np.stack([s.tau.detach().cpu().numpy() for s in scas])
                history["tau"].append(taus[0].tolist())
                history["tau_mean"].append(float(taus.mean()))
                history["empty_row_rate"].append(
                    float(np.mean([s.last_empty_row_rate for s in scas])))
                history["sparsity"].append(
                    float(np.mean([s.last_sparsity for s in scas])))
            if verbose:
                print(f"  ep{ep:03d} train {tot/max(nb,1):.5f} val {vl:.5f}")

            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= tc.patience:
                    break
        if best_state:
            model.load_state_dict(best_state)

    model.eval()
    preds = []
    with torch.no_grad():
        for k in range(0, len(te), 256):
            preds.append(model(t(Xs[te[k:k + 256]])).cpu().numpy())
    pred = np.concatenate(preds) if preds else np.zeros((0, N, H), np.float32)

    # METRICS IN dB - invert the scaling first.
    y_db = sc.inverse(Ys[te])
    p_db = sc.inverse(pred)
    out = dict(
        variant=variant,
        params=int(n_par),
        epochs_run=len(history["val_loss"]),
        seconds=round(time.time() - t_start, 1),
        metrics=M.split_wet_dry(y_db, p_db, wet_flag[te]),
        history=history,
    )
    # Test-set truth and forecasts in dB, so downstream evaluation (the rain-map
    # module) runs against the trained model rather than retraining one.
    out["_arrays"] = dict(x_db=X[te], y_db=y_db, pred_db=p_db, wet=wet_flag[te])

    if variant != "persistence" and getattr(model, "last_attn", None):
        lam = model.last_attn.get("lambda_fwd")
        if lam is not None:
            out["lambda_mean"] = lam.mean((0, 1)).cpu().numpy()
            out["empty_row_rate"] = float(model.last_attn["empty_row_rate"])
            out["sparsity"] = float(model.last_attn["sparsity"])
            pi = model.last_attn.get("pi_fwd")
            if pi is not None:
                out["pi_mean"] = pi.mean(0).cpu().numpy()
        scas = _sca_modules(model)
        if scas:
            out["tau"] = scas[0].tau.detach().cpu().numpy()
    # The fitted scaler travels with the model so it can be checkpointed.
    model.scaler = sc
    return out, model


def run_scenario(cfg: ExperimentConfig, arms=("persistence", "LSTM-ED", "STANN",
                                              "S-BSTN"), out_dir=None, verbose=False):
    """Simulate -> window -> train every arm -> evaluate -> attention recovery."""
    sim = simulate(cfg.sim)
    X, Y, starts, wet_flag, diag = build_dataset(sim, cfg.data)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    res = {"config": cfg.to_dict(), "data_diag": diag, "arms": {}}
    for a in arms:
        r, model = run(a, X, Y, wet_flag, starts, cfg=cfg, verbose=verbose)
        lam = r.pop("lambda_mean", None)
        if lam is not None:
            r["recovery"] = M.attention_recovery(
                lam, sim["lambda_star"], sim["lag_star"],
                horizon_min=cfg.sim.horizon_min)
            if out_dir:
                np.save(Path(out_dir) / f"lambda_{a}.npy", lam)
        pi = r.pop("pi_mean", None)
        if pi is not None and out_dir:
            np.save(Path(out_dir) / f"pi_{a}.npy", pi)
        arrays = r.pop("_arrays", None)
        if arrays is not None and out_dir:
            np.savez_compressed(Path(out_dir) / f"pred_{a}.npz", **arrays)
        tau = r.pop("tau", None)
        if tau is not None:
            r["tau_final"] = tau.tolist()
        res["arms"][a] = r
        if verbose:
            tn = r["metrics"]["wet"].get("TN", {})
            print(f"[{a:12s}] params={r['params']:>7,}  "
                  f"wet RMSE_avg={tn.get('rmse_avg', float('nan')):.4f} dB  "
                  f"({r['seconds']}s)")

    if out_dir:
        p = Path(out_dir); p.mkdir(parents=True, exist_ok=True)
        (p / "results.json").write_text(json.dumps(res, indent=2, default=float))
        np.save(p / "lambda_star.npy", sim["lambda_star"])
        np.save(p / "lag_star.npy", sim["lag_star"])
    return res, sim
