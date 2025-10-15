from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConceptCategories:
    """Concept categories for prompt analysis."""

    character: list[str] = field(
        default_factory=lambda: ["woman", "man", "person", "character"]
    )
    style: list[str] = field(
        default_factory=lambda: ["photorealistic", "anime", "cartoon", "realistic"]
    )
    content: list[str] = field(
        default_factory=lambda: ["portrait", "scene", "landscape"]
    )
    quality: list[str] = field(
        default_factory=lambda: ["masterpiece", "high quality", "detailed"]
    )
