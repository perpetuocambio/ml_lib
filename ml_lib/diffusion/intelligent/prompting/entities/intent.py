"""Intent detection entities."""

from dataclasses import dataclass
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
