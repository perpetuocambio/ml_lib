"""Infrastructure document processing orchestrator - no Domain dependencies."""

from datetime import datetime
from pathlib import Path

from infrastructure.data.extractors.entities.document_processing_result import (
    InfraDocumentProcessingResult,
)
from infrastructure.data.extractors.entities.processing_metadata import (
    InfraProcessingMetadata,
)
from infrastructure.data.extractors.services.entity_extraction_service import (
    InfraEntityExtractionService,
)
from infrastructure.data.extractors.services.text_processing_service import (
    InfraTextProcessingService,
)


class InfraDocumentProcessingOrchestrator:
    """Infrastructure orchestrator for document processing - no Domain dependencies."""

    def __init__(
        self,
        text_processing_service: InfraTextProcessingService,
        entity_extraction_service: InfraEntityExtractionService,
    ):
        """Initialize with Infrastructure services only."""
        self.text_processing_service = text_processing_service
        self.entity_extraction_service = entity_extraction_service

    def process_document_complete(
        self,
        document_id: str,
        file_path: Path,
        processing_method: str,  # Infrastructure uses string instead of enum
    ) -> InfraDocumentProcessingResult:
        """Execute complete document processing pipeline using Infrastructure types."""

        processing_id = f"proc_{document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        started_at = datetime.now()

        # Stage 1: Text Extraction
        text_extraction_result = (
            self.text_processing_service.extract_text_from_document(
                document_id=document_id,
                file_path=file_path,
                processing_method=processing_method,
            )
        )

        # Validate text extraction quality
        if not self.text_processing_service.validate_extraction_quality(
            text_extraction_result
        ):
            raise ValueError(
                f"Text extraction quality validation failed for document {document_id}"
            )

        # Stage 2: Entity Extraction
        extracted_entities = self.entity_extraction_service.extract_entities_from_text(
            text_extraction_result=text_extraction_result,
        )

        # Create processing metadata
        completed_at = datetime.now()
        processing_duration = (completed_at - started_at).total_seconds()

        processing_metadata = InfraProcessingMetadata(
            processing_id=processing_id,
            document_id=document_id,
            stage="PROCESSING_COMPLETE",  # Infrastructure uses string
            started_at=started_at,
            completed_at=completed_at,
            processing_duration_seconds=processing_duration,
        )

        # Create complete result using Infrastructure types
        return InfraDocumentProcessingResult(
            processing_id=processing_id,
            document_id=document_id,
            text_extraction=text_extraction_result,
            extracted_entities=extracted_entities,
            processing_metadata=processing_metadata,
        )

    def validate_processing_prerequisites(self, file_path: Path) -> bool:
        """Validate that document can be processed."""

        if not file_path.exists():
            return False

        # Add more validation as needed
        return file_path.stat().st_size != 0
