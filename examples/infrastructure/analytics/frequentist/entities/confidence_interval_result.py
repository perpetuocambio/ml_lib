"""Confidence interval result entity."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceIntervalResult:
    """Result of confidence interval calculation."""

    lower_bound: float
    upper_bound: float
