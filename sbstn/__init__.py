"""S-BSTN: Selective Bidirectional Spatio-Temporal Network.

Multi-link forecasting of weather-induced attenuation in commercial microwave
link (CML) networks, from the links' own past measurements.

    Jacoby, Messer & Ostrometzky, "Spatio-Temporal Model for Predicting
    Multivariate Weather-Induced Attenuation in Wireless Networks",
    IEEE Transactions on Instrumentation and Measurement, vol. 74, 2025.
"""
__version__ = "1.0.0"

from .models.sbstn import SBSTN, VARIANTS, build  # noqa: E402
from .forecaster import Forecaster  # noqa: E402

__all__ = ["SBSTN", "VARIANTS", "build", "Forecaster", "__version__"]
