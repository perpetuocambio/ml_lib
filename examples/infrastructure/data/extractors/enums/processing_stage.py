"""Processing stage enum."""

from enum import Enum


class ProcessingStage(Enum):
    """Stages of document processing in Phase 2."""

    TEXT_EXTRACTION = "TEXT_EXTRACTION"
    ENTITY_IDENTIFICATION = "ENTITY_IDENTIFICATION"
    CONTENT_STRUCTURING = "CONTENT_STRUCTURING"
    QUALITY_VALIDATION = "QUALITY_VALIDATION"
    PROCESSING_COMPLETE = "PROCESSING_COMPLETE"
