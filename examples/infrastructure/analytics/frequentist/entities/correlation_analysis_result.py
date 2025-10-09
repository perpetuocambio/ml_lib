"""Correlation analysis result."""

from dataclasses import dataclass

from infrastructure.analytics.frequentist.entities.correlation_pair_result import (
    CorrelationPairResult,
)


@dataclass(frozen=True)
class CorrelationAnalysisResult:
    """Result of correlation analysis between variables."""

    analysis_method: str
    correlation_pairs: list[CorrelationPairResult]
    total_variables: int
    significant_correlations_count: int
    significance_threshold: float
