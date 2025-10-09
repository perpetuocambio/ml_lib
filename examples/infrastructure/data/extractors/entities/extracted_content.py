from dataclasses import dataclass, field

from infrastructure.data.extractors.entities.document_metadata import (
    DocumentMetadata,
)
from infrastructure.data.extractors.entities.document_structure import (
    DocumentStructure,
)
from infrastructure.data.extractors.entities.image_info import ImageInfo
from infrastructure.data.extractors.entities.table_info import TableInfo
from infrastructure.data.extractors.enums.extraction_status import (
    ExtractionStatus,
)
from infrastructure.data.extractors.enums.extraction_strategy import (
    ExtractionStrategy,
)


@dataclass
class ExtractedContent:
    """Contenido extraído de un documento."""

    text: str
    metadata: DocumentMetadata
    images: list[ImageInfo] = field(default_factory=list)
    tables: list[TableInfo] = field(default_factory=list)
    structure: DocumentStructure = field(default_factory=DocumentStructure)
    extraction_time: float = 0.0
    extraction_strategy: ExtractionStrategy = ExtractionStrategy.AUTO
    status: ExtractionStatus = ExtractionStatus.SUCCESS
    error_message: str | None = None
    source_file: str | None = None

    @property
    def success(self) -> bool:
        """Compatibilidad hacia atrás - indica si la extracción fue exitosa."""
        return self.status == ExtractionStatus.SUCCESS

    @property
    def is_processing(self) -> bool:
        """Indica si la extracción está en proceso."""
        return self.status == ExtractionStatus.PROCESSING

    @property
    def is_failed(self) -> bool:
        """Indica si la extracción falló."""
        return self.status in {
            ExtractionStatus.FAILED,
            ExtractionStatus.TIMEOUT,
            ExtractionStatus.CANCELLED,
        }
