"""Prompt analysis entities."""

from dataclasses import dataclass, field
from enum import Enum

from ml_lib.diffusion.intelligent.prompting.entities.intent import Intent


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
