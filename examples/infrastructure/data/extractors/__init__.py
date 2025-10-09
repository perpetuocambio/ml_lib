# __init__.py
"""
Módulo de extracción de documentos flexible y extensible.

Este módulo proporciona una interfaz unificada para extraer contenido de diversos
tipos de documentos usando diferentes librerías como Docling y MarkItDown.

Características principales:
- Interfaz extensible para agregar nuevos extractores
- Tipado completo de parámetros
- Detección automática del tipo de documento
- Estrategias de fallback automático
- Extracción de metadatos, imágenes y tablas
- Configuración flexible

Uso básico:
    from document_extractor import extract_document, ExtractionConfig

    # Extracción simple
    result = extract_document("document.pdf")
    print(result.text)

    # Con configuración personalizada
    config = ExtractionConfig(
        strategy=ExtractionStrategy.DOCLING,
        extract_tables=True,
        output_format="markdown"
    )
    result = extract_document("document.docx", config)

Autor: Asistente AI
Versión: 1.0.0
"""

from infrastructure.data.extractors.entities.data_processing_result import (
    DataProcessingResult,
)
from infrastructure.data.extractors.entities.document_metadata import (
    DocumentMetadata,
)
from infrastructure.data.extractors.entities.extracted_content import (
    ExtractedContent,
)

# ExtractionConfig now centralized in infrastructure.config.algorithms.extraction_config
from infrastructure.data.extractors.entities.extraction_errors import (
    ExtractionError,
)
from infrastructure.data.extractors.entities.ingestion_result import (
    IngestionResult,
)
from infrastructure.data.extractors.entities.markdown_export_options import (
    MarkdownExportOptions,
)
from infrastructure.data.extractors.enums.code_block_style import CodeBlockStyle
from infrastructure.data.extractors.enums.document_type import DocumentType
from infrastructure.data.extractors.enums.extraction_strategy import (
    ExtractionStrategy,
)
from infrastructure.data.extractors.enums.table_format import TableFormat
from infrastructure.data.extractors.handlers.content_cleaner import ContentCleaner
from infrastructure.data.extractors.handlers.document_type_detector import (
    DocumentTypeDetector,
)
from infrastructure.data.extractors.handlers.exttractor_registry import (
    ExtractorRegistry,
)
from infrastructure.data.extractors.handlers.file_validator import FileValidator
from infrastructure.data.extractors.interfaces.base_document_extractor import (
    BaseDocumentExtractor,
)
from infrastructure.data.extractors.interfaces.document_extractor import (
    DocumentExtractor,
)
from infrastructure.data.extractors.services.document_ingestion_service import (
    DocumentIngestionService,
)
from infrastructure.data.extractors.services.factory import (
    DocumentExtractorFactory,
)

# Exportar las clases y funciones principales
__all__ = [
    # Types
    "DocumentType",
    "ExtractionStrategy",
    # "ExtractionConfig", # Now centralized
    "TableFormat",
    "CodeBlockStyle",
    "DocumentMetadata",
    "ExtractedContent",
    "ExtractionError",
    "IngestionResult",
    "MarkdownExportOptions",
    "BaseDocumentExtractor",
    "DocumentExtractor",
    # Factory and main interface
    "DocumentExtractorFactory",
    "DocumentIngestionService",
    "ExtractorRegistry",
    # Processing (migrated from processing module)
    "DataProcessingResult",
    "EntityExtractionService",
    "TextProcessingService",
    # Utils
    "DocumentTypeDetector",
    "FileValidator",
    "ContentCleaner",
]

__version__ = "1.0.0"
__author__ = "Assistant AI"
__description__ = "Flexible and extensible document extraction module"
