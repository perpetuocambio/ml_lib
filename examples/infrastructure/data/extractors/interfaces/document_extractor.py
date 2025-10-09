from pathlib import Path
from typing import Protocol, runtime_checkable

from infrastructure.config.algorithms.extraction_config import ExtractionConfig
from infrastructure.data.extractors.entities.extracted_content import (
    ExtractedContent,
)
from infrastructure.data.extractors.entities.extraction_capabilities import (
    ExtractionCapabilities,
)
from infrastructure.data.extractors.enums.document_type import DocumentType


@runtime_checkable
class DocumentExtractor(Protocol):
    """Protocolo que define la interfaz para extractores de documentos."""

    def can_extract(self, file_path: Path, document_type: DocumentType) -> bool:
        """Verifica si el extractor puede procesar el tipo de documento."""
        ...

    def extract(self, file_path: Path, config: ExtractionConfig) -> ExtractedContent:
        """Extrae contenido del documento especificado."""
        ...

    def get_capabilities(self) -> ExtractionCapabilities:
        """Retorna las capacidades del extractor."""
        ...
