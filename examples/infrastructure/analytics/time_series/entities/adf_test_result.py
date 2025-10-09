"""ADF test result entity."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AdfTestResult:
    """Result of Augmented Dickey-Fuller test."""

    test_statistic: float
    p_value: float
    number_of_lags: int
