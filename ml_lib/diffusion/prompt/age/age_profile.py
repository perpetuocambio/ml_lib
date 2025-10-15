from dataclasses import dataclass


@dataclass(frozen=True)
class AgeProfile:
    """Age profile configuration."""

    min_age: int
    max_age: int
    default_weight: float
    probability: float
    features: tuple[str, ...]
