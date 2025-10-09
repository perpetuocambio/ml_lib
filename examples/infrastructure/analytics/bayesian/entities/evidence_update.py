"""Evidence update data entity."""

from dataclasses import dataclass

from infrastructure.analytics.bayesian.entities.hypothesis_mapping import (
    HypothesisMapping,
)


@dataclass
class EvidenceUpdate:
    """New evidence for Bayesian updating."""

    evidence_id: str
    evidence_description: str
    likelihood_mapping: list[HypothesisMapping]
    evidence_weight: float = 1.0

    def get_likelihood_for_hypothesis(self, hypothesis_id: str) -> float:
        """Get likelihood for specific hypothesis."""
        for mapping in self.likelihood_mapping:
            if mapping.hypothesis_id == hypothesis_id:
                return mapping.value
        return 0.5
