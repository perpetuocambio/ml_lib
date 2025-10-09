"""Evidence likelihood mapping entity."""

from dataclasses import dataclass

from infrastructure.analytics.bayesian.entities.hypothesis_mapping import (
    HypothesisMapping,
)


@dataclass
class EvidenceLikelihoodMapping:
    """Maps evidence to hypothesis likelihoods."""

    evidence_id: str
    hypothesis_likelihoods: list[HypothesisMapping]
