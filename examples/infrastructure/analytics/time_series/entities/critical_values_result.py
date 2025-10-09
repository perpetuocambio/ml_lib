"""Critical values result entity."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CriticalValuesResult:
    """Critical values for statistical tests."""

    one_percent: float
    five_percent: float
    ten_percent: float
