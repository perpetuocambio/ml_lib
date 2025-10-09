"""Infrastructure document processing result DTO - no Domain dependencies."""

from dataclasses import dataclass

from infrastructure.data.extractors.entities.extracted_entity import (
    InfraExtractedEntity,
)
from infrastructure.data.extractors.entities.processing_metadata import (
    InfraProcessingMetadata,
)
from infrastructure.data.extractors.entities.text_extraction_result import (
    InfraTextExtractionResult,
)


@dataclass(frozen=True)
class InfraDocumentProcessingResult:
    """Infrastructure DTO for document processing results."""

    processing_id: str
    document_id: str
    text_extraction: InfraTextExtractionResult
    extracted_entities: list[InfraExtractedEntity]
    processing_metadata: InfraProcessingMetadata

    def get_entity_count(self) -> int:
        """Get total number of extracted entities."""
        return len(self.extracted_entities)

    def get_high_confidence_entities(self) -> list[InfraExtractedEntity]:
        """Get entities with high confidence scores."""
        return [
            entity for entity in self.extracted_entities if entity.is_high_confidence()
        ]

    def get_entities_by_type(
        self, entity_type_filter: str
    ) -> list[InfraExtractedEntity]:
        """Get entities filtered by type."""
        return [
            entity
            for entity in self.extracted_entities
            if entity.entity_type == entity_type_filter
        ]

    def get_processing_summary(self) -> str:
        """Get comprehensive processing summary."""
        return (
            f"Processed document {self.document_id}: "
            f"{len(self.text_extraction.extracted_text)} chars extracted, "
            f"{self.get_entity_count()} entities identified "
            f"({len(self.get_high_confidence_entities())} high-confidence)"
        )

    def has_meaningful_results(self) -> bool:
        """Check if processing produced meaningful results."""
        return (
            self.text_extraction.has_meaningful_content()
            and len(self.extracted_entities) > 0
        )
