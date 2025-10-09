"""Individual probability entry for Bayesian calculations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbabilityEntry:
    """Individual probability entry."""

    hypothesis_id: str
    probability: float
