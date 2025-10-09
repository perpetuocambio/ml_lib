"""Statistical hypothesis test result."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HypothesisTestResult:
    """Result of a statistical hypothesis test."""

    test_name: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_statistic: float
    p_value: float
    critical_value: float
    confidence_level: float
    is_significant: bool
    reject_null: bool
    effect_size: float | None = None
    power: float | None = None
