"""
Protocol interfaces for the intelligent diffusion module.

This module defines protocol interfaces to break circular dependencies
and allow proper dependency injection throughout the codebase.

All inline imports should be replaced with these protocol definitions.
"""

from .registry_protocol import ModelRegistryProtocol
from .analyzer_protocol import PromptAnalyzerProtocol
from .recommender_protocol import LoRARecommenderProtocol
from .optimizer_protocol import ParameterOptimizerProtocol
from .memory_protocol import (
    MemoryManagerProtocol,
    ModelOffloaderProtocol,
    MemoryOptimizerProtocol
)
from .learning_protocol import LearningEngineProtocol
from .llm_protocol import LLMClientProtocol

__all__ = [
    "ModelRegistryProtocol",
    "PromptAnalyzerProtocol",
    "LoRARecommenderProtocol",
    "ParameterOptimizerProtocol",
    "MemoryManagerProtocol",
    "ModelOffloaderProtocol",
    "MemoryOptimizerProtocol",
    "LearningEngineProtocol",
    "LLMClientProtocol",
]
