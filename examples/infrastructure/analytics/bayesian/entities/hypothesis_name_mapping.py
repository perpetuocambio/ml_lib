"""Hypothesis name mapping entity."""

from dataclasses import dataclass


@dataclass
class HypothesisNameMapping:
    """Maps hypothesis ID to name."""

    hypothesis_id: str
    hypothesis_name: str
