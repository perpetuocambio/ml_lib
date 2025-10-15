"""Analysis Strategy Interfaces - Strategy Pattern for PromptAnalyzer.

Defines strategy interfaces for different analysis responsibilities:
- Concept Extraction: Extract semantic concepts from prompts
- Intent Detection: Determine artistic style, content type, quality
- Prompt Optimization: Optimize prompts for specific models
- Tokenization: Parse prompts respecting SD syntax

This enables pluggable analysis strategies and better separation of concerns.
"""

from typing import Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum, auto


# Forward declare enums to avoid circular import
class ArtisticStyle(Enum):
    """Artistic style of the prompt."""

    PHOTOREALISTIC = auto()
    ANIME = auto()
    CARTOON = auto()
    PAINTING = auto()
    SKETCH = auto()
    ABSTRACT = auto()
    CONCEPT_ART = auto()


class ContentType(Enum):
    """Type of content in the prompt."""

    CHARACTER = auto()
    PORTRAIT = auto()
    SCENE = auto()
    OBJECT = auto()
    ABSTRACT_CONCEPT = auto()


class QualityLevel(Enum):
    """Quality level for generation."""

    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    MASTERPIECE = auto()


@dataclass(frozen=True)
class ConceptExtractionResult:
    """Result of concept extraction."""

    concepts_by_category: dict[str, list[str]]
    confidence: float  # 0-1


@dataclass(frozen=True)
class IntentDetectionResult:
    """Result of intent detection."""

    artistic_style: ArtisticStyle
    content_type: ContentType
    quality_level: QualityLevel
    confidence: float  # 0-1


@dataclass(frozen=True)
class OptimizationResult:
    """Result of prompt optimization."""

    optimized_prompt: str
    optimized_negative: str
    modifications: list[str]  # Description of changes made


@runtime_checkable
class IConceptExtractionStrategy(Protocol):
    """
    Strategy for extracting concepts from prompts.

    Implementations can use different approaches:
    - Rule-based keyword matching
    - LLM semantic analysis
    - Hybrid approaches
    """

    def extract_concepts(
        self,
        prompt: str,
        tokens: list[str],
    ) -> ConceptExtractionResult:
        """
        Extract concepts from prompt.

        Args:
            prompt: Original prompt text
            tokens: Pre-tokenized prompt

        Returns:
            ConceptExtractionResult with categorized concepts
        """
        ...


@runtime_checkable
class IIntentDetectionStrategy(Protocol):
    """
    Strategy for detecting artistic intent from prompts.

    Determines:
    - Artistic style (photorealistic, anime, etc.)
    - Content type (character, scene, portrait)
    - Quality level (low, medium, high, masterpiece)
    """

    def detect_intent(
        self,
        prompt: str,
        concepts: dict[str, list[str]],
        tokens: list[str],
    ) -> IntentDetectionResult:
        """
        Detect artistic intent from prompt.

        Args:
            prompt: Original prompt text
            concepts: Extracted concepts by category
            tokens: Pre-tokenized prompt

        Returns:
            IntentDetectionResult with style, type, quality
        """
        ...


@runtime_checkable
class IOptimizationStrategy(Protocol):
    """
    Strategy for optimizing prompts for specific models.

    Each model architecture has different requirements:
    - SDXL: quality tags appended, natural language friendly
    - Pony V6: score tags prepended, anatomical negatives
    - SD 1.5: quality tags prepended, conservative weights
    """

    def optimize(
        self,
        prompt: str,
        negative_prompt: str,
        quality_level: QualityLevel,
    ) -> OptimizationResult:
        """
        Optimize prompt for specific model.

        Args:
            prompt: Original positive prompt
            negative_prompt: Original negative prompt
            quality_level: Desired quality level

        Returns:
            OptimizationResult with optimized prompts
        """
        ...

    def get_supported_architecture(self) -> str:
        """
        Get the model architecture this strategy supports.

        Returns:
            Model architecture name (e.g., "SDXL", "Pony V6", "SD 1.5")
        """
        ...


@runtime_checkable
class ITokenizationStrategy(Protocol):
    """
    Strategy for tokenizing prompts.

    Must handle Stable Diffusion syntax:
    - Emphasis: (word) -> 1.1x weight
    - Strong emphasis: ((word)) -> 1.21x weight
    - De-emphasis: [word] -> 0.9x weight
    - Attention: {word} -> preserved
    - Separators: commas, AND keywords
    """

    def tokenize(self, prompt: str) -> list[str]:
        """
        Tokenize prompt respecting SD syntax.

        Args:
            prompt: Prompt to tokenize

        Returns:
            List of tokens
        """
        ...

    def extract_emphasis_map(self, prompt: str) -> dict[str, float]:
        """
        Extract emphasis weights from prompt.

        Args:
            prompt: Prompt with SD emphasis syntax

        Returns:
            Dictionary mapping keywords to emphasis weights
        """
        ...
