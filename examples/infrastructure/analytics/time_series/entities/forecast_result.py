"""Time series forecast result entity."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastResult:
    """Result of time series forecasting."""

    forecast_values: list[float]
    confidence_intervals_lower: list[float]
    confidence_intervals_upper: list[float]
    forecast_horizon: int
    confidence_level: float
    forecast_method: str
    forecast_dates: list[str] | None = None
    prediction_intervals_lower: list[float] | None = None
    prediction_intervals_upper: list[float] | None = None
