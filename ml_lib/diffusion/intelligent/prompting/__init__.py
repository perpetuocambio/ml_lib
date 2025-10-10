"""Intelligent prompting and parameter optimization.

This module provides intelligent character generation with:
- Strong typing (no Dict/dict in public APIs)
- Enums for all limited values
- Clean service-oriented architecture
- Uniform probability selection (fantasy-appropriate)

PUBLIC API:
- Services: CharacterGenerator, LoRARecommender, ParameterOptimizer, etc.
- Models: ValidationResult, GenerationPreferences, GeneratedCharacter
- Enums: SafetyLevel, CharacterFocus, QualityTarget, ComplexityLevel
- Types: AttributeType, AttributeDefinition
"""

# ==================== PUBLIC SERVICES ====================
from ml_lib.diffusion.intelligent.prompting.services import (
    CharacterGenerator,
    LoRARecommender,
    NegativePromptGenerator,
    ParameterOptimizer,
    PromptAnalyzer,
)

# ==================== MODELS ====================
from ml_lib.diffusion.intelligent.prompting.models import (
    ValidationResult,
    CompatibilityMap,
    ConceptMap,
    SelectedAttributes,
    GenerationPreferences,
)

from ml_lib.diffusion.intelligent.prompting.entities import (
    GeneratedCharacter,
)

# ==================== ENUMS ====================
from ml_lib.diffusion.intelligent.prompting.enums import (
    SafetyLevel,
    CharacterFocus,
    QualityTarget,
    ComplexityLevel,
)

# ==================== CORE TYPES ====================
from ml_lib.diffusion.intelligent.prompting.core import (
    AttributeType,
    AttributeDefinition,
)

__all__ = [
    # === PUBLIC SERVICES (Main API) ===
    "CharacterGenerator",
    "LoRARecommender",
    "NegativePromptGenerator",
    "ParameterOptimizer",
    "PromptAnalyzer",

    # === MODELS (Data classes) ===
    "ValidationResult",
    "CompatibilityMap",
    "ConceptMap",
    "SelectedAttributes",
    "GenerationPreferences",
    "GeneratedCharacter",

    # === ENUMS (Configuration) ===
    "SafetyLevel",
    "CharacterFocus",
    "QualityTarget",
    "ComplexityLevel",

    # === CORE TYPES (For advanced usage) ===
    "AttributeType",
    "AttributeDefinition",
]
