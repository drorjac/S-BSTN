"""Public API: Forecaster checkpoint round-trip, predict/attention shapes, CLI."""
import json

import numpy as np
import pytest
import torch

from sbstn import Forecaster, VARIANTS, build
from sbstn.cli import main
from sbstn.data.scaling import MinMax
from sbstn.train import build_dataset_from_array


def _forecaster(N=5, T=8, H=4, variant="S-BSTN"):
    kw = dict(hidden=8, q_e=8, q_d=8)
    torch.manual_seed(0)
    model = build(variant, N, T, H, **kw)
    X = np.random.default_rng(0).normal(40, 3, (32, N, T)).astype("float32")
    return Forecaster(model=model, scaler=MinMax().fit(X), variant=variant,
                      model_kwargs=kw), X


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_checkpoint_round_trip(tmp_path, variant):
    fc, X = _forecaster(variant=variant)
    fc.save(tmp_path / "m.pt")
    fc2 = Forecaster.load(tmp_path / "m.pt")
    assert fc2.variant == variant
    np.testing.assert_allclose(fc.predict(X), fc2.predict(X), atol=1e-5)


def test_predict_shapes_and_units():
    fc, X = _forecaster()
    assert fc.predict(X[0]).shape == (5, 4)
    y = fc.predict(X)
    assert y.shape == (32, 5, 4)
    # outputs are in dB, i.e. on the scale of the inputs, not in [0, 1]
    assert 20 < float(np.median(y)) < 60
    with pytest.raises(ValueError):
        fc.predict(X[:, :3])


def test_attention_maps():
    fc, X = _forecaster()
    a = fc.attention(X)
    assert a["lambda_fwd"].shape == (8, 5, 5)
    np.testing.assert_allclose(a["lambda_fwd"].sum(-1), 1.0, atol=1e-5)
    assert a["pi_fwd"].shape == (4, 8)
    np.testing.assert_allclose(a["pi_fwd"].sum(-1), 1.0, atol=1e-5)
    assert a["tau"].shape == (5,)


def test_real_data_builder_needs_no_ground_truth():
    rng = np.random.default_rng(0)
    x = 40 + 0.3 * np.round(rng.normal(0, 0.3, (4, 30000)) / 0.3)
    x[:, 12000:12600] += np.linspace(0, 8, 600)          # a rain event
    X, Y, starts, wet, diag = build_dataset_from_array(x, _small_data_cfg())
    assert X.shape[1:] == (4, 24) and Y.shape[1:] == (4, 10)
    assert wet.any() and 0 < diag["kept_frac"] < 1


def _small_data_cfg():
    from sbstn.config import DataConfig
    return DataConfig()


def test_cli_train_predict_info(tmp_path, capsys):
    out = tmp_path / "run"
    main(["train", "--config", "configs/quickstart.yaml", "--no-figures",
          "--set", "sim.n_links=4", "sim.duration_min=14400", "train.epochs=1",
          "model.hidden=8", "model.q_e=8", "model.q_d=8", "--out", str(out)])
    assert (out / "sbstn.pt").exists()
    res = json.loads((out / "results.json").read_text())
    assert set(res["arms"]) == {"S-BSTN"}

    x = np.load(out / "test_predictions.npz")["x_db"][:3]
    np.save(tmp_path / "x.npy", x)
    main(["predict", "--checkpoint", str(out / "sbstn.pt"),
          "--input", str(tmp_path / "x.npy"), "--out", str(tmp_path / "y.npy")])
    assert np.load(tmp_path / "y.npy").shape == (3, 4, 10)

    capsys.readouterr()
    main(["info", "--checkpoint", str(out / "sbstn.pt")])
    info = json.loads(capsys.readouterr().out)
    assert info["n_links"] == 4 and info["lead_min"] == 5.0
