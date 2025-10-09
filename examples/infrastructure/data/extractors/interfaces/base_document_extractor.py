import datetime
from abc import ABC, abstractmethod
from pathlib import Path

from infrastructure.config.algorithms.extraction_config import ExtractionConfig
from infrastructure.data.extractors.entities.document_metadata import (
    DocumentMetadata,
)
from infrastructure.data.extractors.entities.document_structure import (
    DocumentStructure,
)
from infrastructure.data.extractors.entities.extracted_content import (
    ExtractedContent,
)
from infrastructure.data.extractors.entities.extraction_capabilities import (
    ExtractionCapabilities,
)
from infrastructure.data.extractors.enums.document_type import DocumentType
from infrastructure.data.extractors.enums.extraction_status import (
    ExtractionStatus,
)
from infrastructure.data.extractors.enums.extraction_strategy import (
    ExtractionStrategy,
)


class BaseDocumentExtractor(ABC):
    """Clase base abstracta para extractores de documentos."""

    def __init__(self, name: str):
        self.name = name
        self.extractor_version = "1.0.0"
        self.is_available = True

    @abstractmethod
    def can_extract(self, file_path: Path, document_type: DocumentType) -> bool:
        """Verifica si el extractor puede procesar el tipo de documento."""
        pass

    @abstractmethod
    def extract(self, file_path: Path, config: ExtractionConfig) -> ExtractedContent:
        """Extrae contenido del documento especificado."""
        pass

    @abstractmethod
    def get_capabilities(self) -> ExtractionCapabilities:
        """Retorna las capacidades del extractor."""
        pass

    def get_supported_types(self) -> list[DocumentType]:
        """Retorna los tipos de documento soportados por este extractor."""
        return self.get_capabilities().supported_types

    def _create_metadata(
        self, file_path: Path, document_type: DocumentType
    ) -> DocumentMetadata:
        """Crea metadatos básicos para el documento."""
        file_stat = file_path.stat()
        return DocumentMetadata(
            file_path=file_path,
            file_size=file_stat.st_size,
            document_type=document_type,
            creation_date=datetime.datetime.fromtimestamp(file_stat.st_ctime),
            modification_date=datetime.datetime.fromtimestamp(file_stat.st_mtime),
        )

    def _create_empty_structure(self) -> DocumentStructure:
        """Crea una estructura de documento vacía."""
        return DocumentStructure()

    def _create_failed_extraction(
        self,
        file_path: Path,
        document_type: DocumentType,
        error_message: str,
        extraction_time: float,
    ) -> ExtractedContent:
        """Crea un resultado de extracción fallido."""
        metadata = self._create_metadata(file_path, document_type)
        return ExtractedContent(
            text="",
            metadata=metadata,
            structure=self._create_empty_structure(),
            extraction_time=extraction_time,
            extraction_strategy=ExtractionStrategy.AUTO,
            status=ExtractionStatus.FAILED,
            error_message=error_message,
        )
