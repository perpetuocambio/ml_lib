"""Prompt analysis entities."""

from dataclasses import dataclass, field
from enum import Enum


class ArtisticStyle(Enum):
    """Detected artistic style."""
    PHOTOREALISTIC = "photorealistic"
    ANIME = "anime"
    CARTOON = "cartoon"
    PAINTING = "painting"
    SKETCH = "sketch"
    ABSTRACT = "abstract"
    CONCEPT_ART = "concept_art"
    UNKNOWN = "unknown"


class ContentType(Enum):
    """Type of content being generated."""
    CHARACTER = "character"
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    SCENE = "scene"
    OBJECT = "object"
    ABSTRACT = "abstract"
    UNKNOWN = "unknown"


class QualityLevel(Enum):
    """Desired quality level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MASTERPIECE = "masterpiece"


@dataclass
class Intent:
    """Detected intent from prompt analysis."""

    artistic_style: ArtisticStyle
    content_type: ContentType
    quality_level: QualityLevel
    confidence: float = 0.0

    def __post_init__(self):
        """Validate intent."""
        if isinstance(self.artistic_style, str):
            self.artistic_style = ArtisticStyle(self.artistic_style)
        if isinstance(self.content_type, str):
            self.content_type = ContentType(self.content_type)
        if isinstance(self.quality_level, str):
            self.quality_level = QualityLevel(self.quality_level)

        assert 0.0 <= self.confidence <= 1.0, "Confidence must be between 0 and 1"


class Priority(Enum):
    """Optimization priority."""
    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"


@dataclass
class OptimizedParameters:
    """Optimized generation parameters."""

    num_steps: int
    guidance_scale: float
    width: int
    height: int
    sampler_name: str
    clip_skip: int = 1

    # Estimations
    estimated_time_seconds: float = 0.0
    estimated_vram_gb: float = 0.0
    estimated_quality_score: float = 0.0

    # Strategy
    optimization_strategy: str = "balanced"
    confidence: float = 0.85

    def __post_init__(self):
        """Validate parameters."""
        assert 1 <= self.num_steps <= 150, "Steps must be between 1 and 150"
        assert 1.0 <= self.guidance_scale <= 30.0, "CFG must be between 1 and 30"
        assert self.width > 0 and self.height > 0, "Dimensions must be positive"
        assert 0 <= self.clip_skip <= 12, "Clip skip must be between 0 and 12"

    @property
    def resolution(self) -> tuple[int, int]:
        """Get resolution as tuple."""
        return (self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio."""
        return self.width / self.height if self.height > 0 else 1.0


class ComplexityCategory(Enum):
    """Complexity category for prompts."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class PromptAnalysis:
    """Result of prompt analysis."""

    original_prompt: str
    tokens: list[str] = field(default_factory=list)
    detected_concepts: dict[str, list[str]] = field(default_factory=dict)
    intent: Intent | None = None
    complexity_score: float = 0.0
    emphasis_map: dict[str, float] = field(default_factory=dict)

    @property
    def complexity_category(self) -> ComplexityCategory:
        """Get complexity category from score."""
        if self.complexity_score < 0.3:
            return ComplexityCategory.SIMPLE
        elif self.complexity_score < 0.7:
            return ComplexityCategory.MODERATE
        else:
            return ComplexityCategory.COMPLEX

    @property
    def concept_count(self) -> int:
        """Total number of detected concepts."""
        return sum(len(concepts) for concepts in self.detected_concepts.values())

    def get_concepts_by_category(self, category: str) -> list[str]:
        """Get concepts for a specific category."""
        return self.detected_concepts.get(category, [])

    def has_concept(self, concept: str) -> bool:
        """Check if a concept is present."""
        for concepts in self.detected_concepts.values():
            if concept.lower() in [c.lower() for c in concepts]:
                return True
        return False
