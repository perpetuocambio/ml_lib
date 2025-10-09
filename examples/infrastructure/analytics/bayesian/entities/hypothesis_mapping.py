"""Hypothesis mapping entity."""

from dataclasses import dataclass


@dataclass
class HypothesisMapping:
    """Maps hypothesis ID to a value."""

    hypothesis_id: str
    value: float
