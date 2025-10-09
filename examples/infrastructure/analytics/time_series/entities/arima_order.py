"""ARIMA model order parameters."""

from dataclasses import dataclass


@dataclass
class ARIMAOrder:
    """ARIMA model order parameters."""

    p: int  # Autoregressive order
    d: int  # Differencing order
    q: int  # Moving average order
