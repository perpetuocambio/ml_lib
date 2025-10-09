"""Frequentist analysis entities."""

from infrastructure.analytics.frequentist.entities.confidence_interval_result import (
    ConfidenceIntervalResult,
)
from infrastructure.analytics.frequentist.entities.correlation_analysis_result import (
    CorrelationAnalysisResult,
)
from infrastructure.analytics.frequentist.entities.correlation_pair_result import (
    CorrelationPairResult,
)
from infrastructure.analytics.frequentist.entities.descriptive_statistics_result import (
    DescriptiveStatisticsResult,
)
from infrastructure.analytics.frequentist.entities.hypothesis_test_result import (
    HypothesisTestResult,
)
from infrastructure.analytics.frequentist.entities.statistical_data_input import (
    StatisticalDataInput,
)

__all__ = [
    "ConfidenceIntervalResult",
    "CorrelationAnalysisResult",
    "CorrelationPairResult",
    "DescriptiveStatisticsResult",
    "HypothesisTestResult",
    "StatisticalDataInput",
]
