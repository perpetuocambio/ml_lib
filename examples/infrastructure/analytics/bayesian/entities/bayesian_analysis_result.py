"""Bayesian analysis result entity."""

from dataclasses import dataclass

from infrastructure.analytics.bayesian.entities.evidence_update import EvidenceUpdate
from infrastructure.analytics.bayesian.entities.hypothesis_probability import (
    HypothesisProbability,
)


@dataclass
class BayesianAnalysisResult:
    """Complete Bayesian analysis result."""

    analysis_id: str
    hypotheses: list[HypothesisProbability]
    evidence_used: list[EvidenceUpdate]
    total_evidence_count: int
    most_probable_hypothesis: HypothesisProbability
    probability_distribution_entropy: float
    convergence_status: str
