from dataclasses import dataclass


@dataclass(frozen=True)
class AttributeConfig:
    """Base configuration for character attributes."""

    keywords: tuple[str, ...]
    probability: float = 1.0
    min_age: int = 18
    max_age: int = 80
