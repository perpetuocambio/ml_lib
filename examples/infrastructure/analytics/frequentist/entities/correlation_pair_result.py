"""Correlation result between two variables."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CorrelationPairResult:
    """Correlation analysis result for a pair of variables."""

    variable_x: str
    variable_y: str
    correlation_coefficient: float
    p_value: float
    is_significant: bool
    confidence_interval_lower: float
    confidence_interval_upper: float
