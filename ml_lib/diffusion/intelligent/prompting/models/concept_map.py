"""Concept map model - replaces dict[str, list[str]] from prompt analyzer."""

from dataclasses import dataclass, field


@dataclass
class ConceptMap:
    """Map of detected concepts from prompt analysis.

    This class replaces dict[str, list[str]] that was previously used to store
    detected concepts in PromptAnalyzer.

    All concept categories are strongly typed fields instead of dictionary keys.
    """

    anatomy: list[str] = field(default_factory=list)
    """Anatomical concepts (body parts, physical features)."""

    activity: list[str] = field(default_factory=list)
    """Activity concepts (poses, actions, interactions)."""

    quality: list[str] = field(default_factory=list)
    """Quality modifiers (detailed, hyperrealistic, 8k, etc.)."""

    clothing: list[str] = field(default_factory=list)
    """Clothing and outfit concepts."""

    subjects: list[str] = field(default_factory=list)
    """Subject concepts (person, character, group size, etc.)."""

    age_attributes: list[str] = field(default_factory=list)
    """Age-related attributes and descriptors."""

    physical_details: list[str] = field(default_factory=list)
    """Physical detail concepts (skin texture, pores, etc.)."""

    environment: list[str] = field(default_factory=list)
    """Environment and setting concepts."""

    lighting: list[str] = field(default_factory=list)
    """Lighting-related concepts."""

    style: list[str] = field(default_factory=list)
    """Artistic style concepts."""

    @property
    def is_empty(self) -> bool:
        """Whether this concept map has any concepts."""
        return not any([
            self.anatomy,
            self.activity,
            self.quality,
            self.clothing,
            self.subjects,
            self.age_attributes,
            self.physical_details,
            self.environment,
            self.lighting,
            self.style,
        ])

    @property
    def total_concept_count(self) -> int:
        """Total number of concepts across all categories."""
        return sum([
            len(self.anatomy),
            len(self.activity),
            len(self.quality),
            len(self.clothing),
            len(self.subjects),
            len(self.age_attributes),
            len(self.physical_details),
            len(self.environment),
            len(self.lighting),
            len(self.style),
        ])

    def get_all_concepts(self) -> list[str]:
        """Get all concepts from all categories as a flat list."""
        return (
            self.anatomy
            + self.activity
            + self.quality
            + self.clothing
            + self.subjects
            + self.age_attributes
            + self.physical_details
            + self.environment
            + self.lighting
            + self.style
        )
