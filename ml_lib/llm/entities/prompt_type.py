"""
Tipos de prompts para diferentes fases del análisis.
"""

from enum import Enum


class PromptType(Enum):
    """Tipos de prompts para diferentes fases del análisis."""

    DATA_PREPROCESSING = "data_preprocessing"
    PATTERN_RECOGNITION = "pattern_recognition"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    EVIDENCE_ANALYSIS = "evidence_analysis"
    RESULT_SYNTHESIS = "result_synthesis"
    RECOMMENDATION_GENERATION = "recommendation_generation"
    VALIDATION = "validation"
