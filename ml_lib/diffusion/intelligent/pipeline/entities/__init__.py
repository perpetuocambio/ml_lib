"""Pipeline entities for intelligent generation."""

from .pipeline_config import (
    PipelineConfig,
    OperationMode,
    GenerationConstraints,
    LoRAPreferences,
    MemorySettings,
    Priority,
)
from .generation_result import (
    GenerationResult,
    GenerationMetadata,
    LoRAInfo,
)
from .metadata_dict import (
    GenerationMetadataSerializable,
    LoRASerializable,
)
from .generation_explanation import GenerationExplanation
from .recommendations import Recommendations
from .batch_config import (
    BatchConfig,
    VariationStrategy,
)

__all__ = [
    # Config
    "PipelineConfig",
    "OperationMode",
    "GenerationConstraints",
    "LoRAPreferences",
    "MemorySettings",
    "Priority",
    # Results
    "GenerationResult",
    "GenerationMetadata",
    "LoRAInfo",
    # Serialization
    "GenerationMetadataSerializable",
    "LoRASerializable",
    # Explanation
    "GenerationExplanation",
    # Recommendations
    "Recommendations",
    # Batch
    "BatchConfig",
    "VariationStrategy",
]
