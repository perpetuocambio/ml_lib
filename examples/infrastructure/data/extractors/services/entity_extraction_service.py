"""Infrastructure entity extraction service - no Domain dependencies."""

from infrastructure.data.extractors.entities.entity_match import (
    InfraEntityMatch,
)
from infrastructure.data.extractors.entities.extracted_entity import (
    InfraExtractedEntity,
)
from infrastructure.data.extractors.entities.text_extraction_result import (
    InfraTextExtractionResult,
)


class InfraEntityExtractionService:
    """Infrastructure service for extracting entities from processed text."""

    def extract_entities_from_text(
        self,
        text_extraction_result: InfraTextExtractionResult,
    ) -> list[InfraExtractedEntity]:
        """Extract entities from processed text."""

        extracted_entities = []
        text = text_extraction_result.extracted_text

        # Simple placeholder entity extraction
        # In real implementation, this would use NLP libraries or LLM services
        entities_found = self._find_placeholder_entities(text)

        for i, entity_match in enumerate(entities_found):
            confidence_score = 0.75  # Infrastructure uses simple float

            entity = InfraExtractedEntity(
                entity_id=f"{text_extraction_result.document_id}_entity_{i}",
                entity_text=entity_match.entity_text,
                entity_type=entity_match.entity_type,  # Already a string in Infra
                context_snippet=self._extract_context_snippet(
                    text, entity_match.start_position, entity_match.end_position
                ),
                start_position=entity_match.start_position,
                end_position=entity_match.end_position,
                confidence_score=confidence_score,
                source_document_id=text_extraction_result.document_id,
            )
            extracted_entities.append(entity)

        return extracted_entities

    def _find_placeholder_entities(self, text: str) -> list[InfraEntityMatch]:
        """Find placeholder entities in text (simplified implementation)."""

        entities = []

        # Look for simple patterns (placeholder implementation)
        words = text.split()
        position = 0

        for word in words:
            start_pos = position
            end_pos = position + len(word)

            # Simple heuristics for entity types
            if word.isupper() and len(word) > 2:
                entities.append(
                    InfraEntityMatch(
                        entity_text=word,
                        entity_type="ORGANIZATION",  # Infrastructure uses string
                        start_position=start_pos,
                        end_position=end_pos,
                    )
                )
            elif word.istitle() and len(word) > 3:
                entities.append(
                    InfraEntityMatch(
                        entity_text=word,
                        entity_type="PERSON",  # Infrastructure uses string
                        start_position=start_pos,
                        end_position=end_pos,
                    )
                )

            position = end_pos + 1  # +1 for space

        return entities

    def _extract_context_snippet(self, text: str, start_pos: int, end_pos: int) -> str:
        """Extract context snippet around entity."""

        context_window = 50
        context_start = max(0, start_pos - context_window)
        context_end = min(len(text), end_pos + context_window)

        return text[context_start:context_end]

    def filter_high_confidence_entities(
        self, entities: list[InfraExtractedEntity]
    ) -> list[InfraExtractedEntity]:
        """Filter entities by confidence threshold."""
        confidence_threshold = 0.7
        return [
            entity
            for entity in entities
            if entity.confidence_score >= confidence_threshold
        ]
