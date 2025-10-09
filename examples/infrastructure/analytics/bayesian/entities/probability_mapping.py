"""Probability mapping for Bayesian calculations."""

from dataclasses import dataclass

from infrastructure.analytics.bayesian.entities.probability_entry import (
    ProbabilityEntry,
)


@dataclass(frozen=True)
class ProbabilityMapping:
    """Type-safe container for probability mappings - replaces dict mappings via ProtocolSerializer."""

    entries: list[ProbabilityEntry]

    def get_probability(self, hypothesis_id: str) -> float | None:
        """Get probability for hypothesis."""
        for entry in self.entries:
            if entry.hypothesis_id == hypothesis_id:
                return entry.probability
        return None

    def get_all_ids(self) -> list[str]:
        """Get all hypothesis IDs."""
        return [entry.hypothesis_id for entry in self.entries]

    def items(self) -> list[tuple[str, float]]:
        """Get all entries as tuples for compatibility."""
        return [(entry.hypothesis_id, entry.probability) for entry in self.entries]
