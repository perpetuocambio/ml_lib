"""Descriptive statistics analysis result."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DescriptiveStatisticsResult:
    """Result of descriptive statistical analysis."""

    variable_name: str
    count: int
    mean: float
    median: float
    std_deviation: float
    variance: float
    minimum: float
    maximum: float
    range_value: float
    q1: float
    q3: float
    iqr: float
    skewness: float
    kurtosis: float
    coefficient_of_variation: float
