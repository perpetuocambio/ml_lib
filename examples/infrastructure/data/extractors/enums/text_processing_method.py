"""Text processing method enum."""

from enum import Enum


class TextProcessingMethod(Enum):
    """Methods for processing text in documents."""

    DOCLING_OCR = "DOCLING_OCR"
    MARKITDOWN_EXTRACTION = "MARKITDOWN_EXTRACTION"
    HYBRID_PROCESSING = "HYBRID_PROCESSING"
    LLM_ASSISTED_PARSING = "LLM_ASSISTED_PARSING"
    STRUCTURED_EXTRACTION = "STRUCTURED_EXTRACTION"
