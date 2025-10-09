"""Infrastructure text processing service - no Domain dependencies."""

from datetime import datetime
from pathlib import Path

from infrastructure.data.extractors.entities.text_extraction_result import (
    InfraTextExtractionResult,
)


class InfraTextProcessingService:
    """Infrastructure service for extracting and processing text from documents."""

    def extract_text_from_document(
        self,
        document_id: str,
        file_path: Path,
        processing_method: str,  # Infrastructure uses string instead of enum
    ) -> InfraTextExtractionResult:
        """Extract text from a document using specified method."""

        # For now, this is a placeholder that would integrate with infrastructure extractors
        # In real implementation, this would delegate to infrastructure services
        extracted_text = (
            f"Extracted text from {file_path.name} using {processing_method}"
        )

        confidence_score = 0.85  # Infrastructure uses simple float

        return InfraTextExtractionResult(
            document_id=document_id,
            source_file_path=file_path,
            extracted_text=extracted_text,
            extraction_method=processing_method,  # Already a string
            confidence_score=confidence_score,
            extraction_timestamp=datetime.now(),
        )

    def validate_extraction_quality(
        self, extraction_result: InfraTextExtractionResult
    ) -> bool:
        """Validate the quality of text extraction."""

        # Check minimum text length
        if not extraction_result.has_meaningful_content():
            return False

        # Check confidence threshold
        return extraction_result.confidence_score.score >= 0.7
