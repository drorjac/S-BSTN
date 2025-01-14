"""ITU-R P.838-3 specific attenuation coefficients - loader and validator.

    gamma = k * R^alpha        [dB/km],  R in mm/h

WHY THIS FILE CONTAINS NO NUMBERS
---------------------------------
The coefficient tables must be taken from the recommendation itself. Getting
them wrong silently biases every rain number downstream, so they are not
bundled. This module supplies everything around them - the P.838-3 functional
form, the polarisation/elevation adjustment, the loader, and the sanity checks -
and raises a clear error until `itu_r_838_coeffs.json`, transcribed from the
recommendation, is placed next to it.

Until then the simulator runs on the placeholder law (SimConfig.pl_k = 0.1,
pl_alpha = 1.0): every rain-rate quantity it produces is internally consistent
but not physically calibrated. Forecasting in dB is unaffected.

EXPECTED FILE FORMAT
--------------------
`itu_r_838_coeffs.json`, next to this file:

    {
      "source": "ITU-R P.838-3, Tables 1-4",
      "kH": {"a": [...4 values...], "b": [...], "c": [...], "m": <float>, "c0": <float>},
      "kV": {...}, "aH": {...}, "aV": {...}
    }

matching the recommendation's fitted form, with j running over the 4 (kH, kV)
or 5 (aH, aV) Gaussian terms:

    log10 k = SUM_j a_j * exp(-((log10 f - b_j) / c_j)^2) + m * log10 f + c0
    alpha   = SUM_j a_j * exp(-((log10 f - b_j) / c_j)^2) + m * log10 f + c0
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

_COEFF_PATH = Path(__file__).with_name("itu_r_838_coeffs.json")

_MISSING = (
    "ITU-R P.838-3 coefficients are not present.\n"
    f"Expected: {_COEFF_PATH}\n\n"
    "They are deliberately not bundled: transcribe them from the "
    "recommendation itself. Until then the simulator uses the "
    "placeholder law in SimConfig (pl_k, pl_alpha), and no quantitative mm/h "
    "claim is supportable.\n\n"
    "To enable: transcribe Tables 1-4 of P.838-3 into that JSON file, then run "
    "`python -c 'from sbstn.data.itu_r_838 import check_monotonicity as c; print(c())'`."
)


def coefficients_available() -> bool:
    return _COEFF_PATH.exists()


def _load() -> dict:
    if not _COEFF_PATH.exists():
        raise FileNotFoundError(_MISSING)
    return json.loads(_COEFF_PATH.read_text())


def _fit(entry: dict, f_ghz: np.ndarray) -> np.ndarray:
    """The recommendation's sum-of-Gaussians fit in log10 f."""
    lf = np.log10(f_ghz)
    a = np.asarray(entry["a"])[:, None]
    b = np.asarray(entry["b"])[:, None]
    c = np.asarray(entry["c"])[:, None]
    terms = a * np.exp(-(((lf[None, :] - b) / c) ** 2))
    return terms.sum(0) + entry["m"] * lf + entry["c0"]


def k_alpha(f_ghz, polarization: str = "V", elevation_deg: float = 0.0,
            tilt_deg: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Specific-attenuation coefficients for the given links.

    `tilt_deg` defaults to 0 for horizontal and 90 for vertical polarisation,
    which is the usual case for terrestrial CMLs.
    """
    co = _load()
    f = np.atleast_1d(np.asarray(f_ghz, dtype=float))
    kH = 10.0 ** _fit(co["kH"], f)
    kV = 10.0 ** _fit(co["kV"], f)
    aH = _fit(co["aH"], f)
    aV = _fit(co["aV"], f)

    if tilt_deg is None:
        tilt_deg = 90.0 if polarization.upper().startswith("V") else 0.0
    th = np.deg2rad(elevation_deg)
    ta = np.deg2rad(tilt_deg)

    # P.838-3 polarisation/elevation combination.
    k = (kH + kV + (kH - kV) * np.cos(th) ** 2 * np.cos(2 * ta)) / 2.0
    alpha = (kH * aH + kV * aV
             + (kH * aH - kV * aV) * np.cos(th) ** 2 * np.cos(2 * ta)) / (2.0 * k)
    return k, alpha


def check_monotonicity() -> dict:
    """Sanity properties that must hold regardless of source:

    k rises monotonically with frequency over 10-40 GHz, alpha falls toward ~1,
    and horizontal k exceeds vertical k at equal frequency.
    """
    co = _load()
    f = np.linspace(10.0, 40.0, 61)
    kH = 10.0 ** _fit(co["kH"], f)
    kV = 10.0 ** _fit(co["kV"], f)
    aH = _fit(co["aH"], f)
    return dict(
        k_increasing=bool(np.all(np.diff(kH) > 0) and np.all(np.diff(kV) > 0)),
        alpha_decreasing=bool(np.all(np.diff(aH) < 0)),
        kH_exceeds_kV=bool(np.all(kH > kV)),
    )


# ----------------------------------------------------------------------
def attenuation_from_rain(r_mmh, length_km, k, alpha):
    """gamma = k R^alpha  [dB/km];  A = gamma * L  [dB]."""
    r = np.maximum(np.asarray(r_mmh, dtype=float), 0.0)
    return k * np.power(r, alpha) * length_km


def rain_from_attenuation(A_dB, length_km, k, alpha):
    """Invert the power law: R = (A / (L k))^(1/alpha).

    This is step 2 of the rain-map pipeline in sbstn/evaluate.py.
    Negative attenuation (noise on a dry link) clips to zero rain.
    """
    gamma = np.maximum(np.asarray(A_dB, dtype=float), 0.0) / np.maximum(length_km, 1e-9)
    return np.power(gamma / k, 1.0 / alpha)
