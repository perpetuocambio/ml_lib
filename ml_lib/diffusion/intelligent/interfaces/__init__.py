"""
Protocol interfaces for the intelligent diffusion module.

This module defines protocol interfaces to break circular dependencies
and allow proper dependency injection throughout the codebase.

All inline imports should be replaced with these protocol definitions.
"""

from ml_lib.diffusion.intelligent.interfaces.registry_protocol import ModelRegistryProtocol
from ml_lib.diffusion.intelligent.interfaces.analyzer_protocol import PromptAnalyzerProtocol
from ml_lib.diffusion.intelligent.interfaces.recommender_protocol import LoRARecommenderProtocol
from ml_lib.diffusion.intelligent.interfaces.optimizer_protocol import ParameterOptimizerProtocol
from ml_lib.diffusion.intelligent.interfaces.memory_protocol import (
    MemoryManagerProtocol,
    ModelOffloaderProtocol,
    MemoryOptimizerProtocol
)
from ml_lib.diffusion.intelligent.interfaces.learning_protocol import LearningEngineProtocol
from ml_lib.diffusion.intelligent.interfaces.llm_protocol import LLMClientProtocol

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
