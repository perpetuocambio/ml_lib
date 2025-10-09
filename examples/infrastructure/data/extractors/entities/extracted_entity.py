"""Infrastructure extracted entity DTO - no Domain dependencies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class InfraExtractedEntity:
    """Infrastructure DTO for extracted entities."""

    entity_id: str
    entity_text: str
    entity_type: str  # Infrastructure uses string instead of enum
    context_snippet: str
    start_position: int
    end_position: int
    confidence_score: float  # Infrastructure uses float instead of ConfidenceMetric
    source_document_id: str

    def get_entity_span(self) -> int:
        """Get the span length of the entity in the text."""
        return self.end_position - self.start_position

    def is_high_confidence(self) -> bool:
        """Check if entity extraction has high confidence."""
        return self.confidence_score >= 0.8

    def get_entity_summary(self) -> str:
        """Get a summary of the extracted entity."""
        return (
            f"{self.entity_type}: '{self.entity_text}' "
            f"(confidence: {self.confidence_score:.2f}) "
            f"at position {self.start_position}-{self.end_position}"
        )
