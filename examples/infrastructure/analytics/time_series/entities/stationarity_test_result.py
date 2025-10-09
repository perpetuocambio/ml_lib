"""Stationarity test result entity."""

from dataclasses import dataclass

from infrastructure.analytics.time_series.entities.critical_values_result import (
    CriticalValuesResult,
)


@dataclass(frozen=True)
class StationarityTestResult:
    """Result of stationarity testing."""

    test_name: str
    test_statistic: float
    p_value: float
    critical_values: CriticalValuesResult
    is_stationary: bool
    confidence_level: float
    number_of_lags: int | None = None
    trend_component: str | None = None
