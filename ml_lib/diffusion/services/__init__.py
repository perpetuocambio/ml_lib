"""Diffusion services."""

from .intelligent_pipeline import IntelligentGenerationPipeline
from .batch_processor import BatchProcessor
from .decision_explainer import DecisionExplainer, ExplanationVerbosity, DecisionContext
from .feedback_collector import (
    FeedbackCollector,
    GenerationSession,
    UserFeedback,
)
from .intelligent_builder import (
    IntelligentPipelineBuilder,
    GenerationConfig,
    SelectedModels,
)
from .model_orchestrator import (
    ModelOrchestrator,
    ModelMetadataFile,
    DiffusionArchitecture,
)
from .ollama_selector import (
    OllamaModelSelector,
    ModelMatcher,
    PromptAnalysis,
)
from .image_naming import ImageNamingConfig
from .image_metadata import (
    ImageMetadataWriter,
    ImageMetadataEmbedding,
    create_generation_id,
    create_timestamp,
)
from .character_generator import CharacterGenerator
from .learning_engine import LearningEngine
from .lora_recommender import LoRARecommender
from .negative_prompt_generator import NegativePromptGenerator
from .parameter_optimizer import ParameterOptimizer
from .prompt_analyzer import PromptAnalyzer
from .memory_optimizer import (
    MemoryOptimizer,
    MemoryOptimizationConfig,
    OptimizationLevel,
    MemoryMonitor,
)
from .metadata_fetcher import MetadataFetcher, ModelMetadata, FetcherConfig

__all__ = [
    # Pipeline
    "IntelligentGenerationPipeline",
    # Batch processing
    "BatchProcessor",
    # Explanation
    "DecisionExplainer",
    "ExplanationVerbosity",
    "DecisionContext",
    # Feedback
    "FeedbackCollector",
    "GenerationSession",
    "UserFeedback",
    # Builder
    "IntelligentPipelineBuilder",
    "GenerationConfig",
    "SelectedModels",
    # Orchestration
    "ModelOrchestrator",
    "ModelMetadataFile",
    "DiffusionArchitecture",
    # Ollama selector
    "OllamaModelSelector",
    "ModelMatcher",
    "PromptAnalysis",
    # Metadata
    "ImageMetadataWriter",
    "ImageMetadataEmbedding",
    "ImageNamingConfig",
    "create_generation_id",
    "create_timestamp",
    # Prompting services
    "CharacterGenerator",
    "LearningEngine",
    "LoRARecommender",
    "NegativePromptGenerator",
    "ParameterOptimizer",
    "PromptAnalyzer",
    # Memory optimization
    "MemoryOptimizer",
    "MemoryOptimizationConfig",
    "OptimizationLevel",
    "MemoryMonitor",
    # Metadata fetching
    "MetadataFetcher",
    "ModelMetadata",
    "FetcherConfig",
]
