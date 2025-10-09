"""Confidence interval for Bayesian calculations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceInterval:
    """Type-safe confidence interval - replaces tuple[float, float]."""

    lower_bound: float
    upper_bound: float

    def get_range(self) -> float:
        """Get interval range."""
        return self.upper_bound - self.lower_bound

    def contains(self, value: float) -> bool:
        """Check if value is within interval."""
        return self.lower_bound <= value <= self.upper_bound
