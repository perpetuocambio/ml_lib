"""Domain Strategies - Strategy Pattern implementations.

This package contains concrete strategy implementations for:
- Concept Extraction: Rule-based, LLM-enhanced, Hybrid
- Intent Detection: Rule-based, LLM-enhanced
- Tokenization: Simple, StableDiffusion, Advanced
- Prompt Optimization: SDXL, Pony V6, SD 1.5
"""

from ml_lib.diffusion.domain.strategies.concept_extraction import (
    RuleBasedConceptExtraction,
    LLMEnhancedConceptExtraction,
    HybridConceptExtraction,
)
from ml_lib.diffusion.domain.strategies.intent_detection import (
    RuleBasedIntentDetection,
    LLMEnhancedIntentDetection,
)
from ml_lib.diffusion.domain.strategies.tokenization import (
    SimpleTokenization,
    StableDiffusionTokenization,
    AdvancedTokenization,
)
from ml_lib.diffusion.domain.strategies.optimization import (
    SDXLOptimizationStrategy,
    PonyV6OptimizationStrategy,
    SD15OptimizationStrategy,
    OptimizationStrategyFactory,
)

__all__ = [
    # Concept Extraction
    "RuleBasedConceptExtraction",
    "LLMEnhancedConceptExtraction",
    "HybridConceptExtraction",
    # Intent Detection
    "RuleBasedIntentDetection",
    "LLMEnhancedIntentDetection",
    # Tokenization
    "SimpleTokenization",
    "StableDiffusionTokenization",
    "AdvancedTokenization",
    # Optimization
    "SDXLOptimizationStrategy",
    "PonyV6OptimizationStrategy",
    "SD15OptimizationStrategy",
    "OptimizationStrategyFactory",
]
