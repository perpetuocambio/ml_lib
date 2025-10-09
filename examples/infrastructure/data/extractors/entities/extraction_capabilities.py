# types.py
from dataclasses import dataclass

from infrastructure.data.extractors.enums.document_type import DocumentType


@dataclass
class ExtractionCapabilities:
    """Capacidades de un extractor."""

    supported_types: list[DocumentType]
    can_extract_images: bool = False
    can_extract_tables: bool = False
    can_extract_metadata: bool = False
    can_preserve_formatting: bool = False
    supports_ocr: bool = False
    max_file_size_mb: int = 100
