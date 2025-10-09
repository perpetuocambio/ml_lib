import logging
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from infrastructure.config.algorithms.extraction_config import ExtractionConfig
from infrastructure.data.extractors.entities.docling_config import DoclingConfig
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
from infrastructure.data.extractors.entities.extraction_errors import (
    ExtractionError,
)
from infrastructure.data.extractors.entities.image_info import ImageInfo
from infrastructure.data.extractors.entities.structural_element import (
    StructuralElement,
)
from infrastructure.data.extractors.entities.table_info import TableInfo
from infrastructure.data.extractors.enums.document_type import DocumentType
from infrastructure.data.extractors.enums.extraction_status import (
    ExtractionStatus,
)
from infrastructure.data.extractors.enums.extraction_strategy import (
    ExtractionStrategy,
)
from infrastructure.data.extractors.enums.processing_priority import (
    ProcessingPriority,
)
from infrastructure.data.extractors.handlers.content_cleaner import ContentCleaner
from infrastructure.data.extractors.handlers.document_type_detector import (
    DocumentTypeDetector,
)
from infrastructure.data.extractors.interfaces.base_document_extractor import (
    BaseDocumentExtractor,
)
from infrastructure.serialization.protocol_serializer import ProtocolSerializer

logger = logging.getLogger(__name__)


class DoclingExtractor(BaseDocumentExtractor):
    """Extractor de documentos usando la librería Docling."""

    # Mapeo de tipos de documento a formatos de entrada de Docling
    FORMAT_MAP = ProtocolSerializer.serialize_format_mapping(
        {
            DocumentType.PDF: InputFormat.PDF,
            DocumentType.DOCX: InputFormat.DOCX,
            DocumentType.PPTX: InputFormat.PPTX,
            DocumentType.HTML: InputFormat.HTML,
            DocumentType.MD: InputFormat.MD,
        }
    )

    def __init__(self, document_type_detector: DocumentTypeDetector):
        super().__init__("DoclingExtractor")
        self._converter = None
        self.document_type_detector = document_type_detector

    @property
    def converter(self) -> DocumentConverter:
        """Lazy loading del converter de Docling."""
        if self._converter is None:
            # Configurar opciones de pipeline para PDF
            pdf_options = PdfPipelineOptions(
                do_ocr=True,
                do_table_structure=True,
                table_structure_options={
                    "do_cell_matching": True,
                },
            )

            # Crear converter con configuraciones optimizadas
            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
            }

            self._converter = DocumentConverter(format_options=format_options)

        return self._converter

    def can_extract(self, file_path: Path, document_type: DocumentType) -> bool:
        """Verifica si Docling puede procesar el tipo de documento."""
        return document_type in self.FORMAT_MAP

    def get_supported_types(self) -> list[DocumentType]:
        """Retorna los tipos de documento soportados por Docling."""
        return list(self.FORMAT_MAP.keys())

    def get_capabilities(self) -> ExtractionCapabilities:
        """Retorna las capacidades del extractor Docling."""
        return ExtractionCapabilities(
            supported_types=self.get_supported_types(),
            can_extract_images=True,
            can_extract_tables=True,
            can_extract_metadata=True,
            can_preserve_formatting=True,  # Docling preserves formatting by default
            supports_ocr=True,
            max_file_size_mb=100,  # Default max file size
        )

    def extract(self, file_path: Path, config: ExtractionConfig) -> ExtractedContent:
        """
        Extrae contenido usando Docling.

        Args:
            file_path: Ruta al archivo a procesar
            config: Configuración de extracción

        Returns:
            ExtractedContent: Contenido extraído
        """
        start_time = time.time()

        try:
            # Detectar tipo de documento
            document_type = self.document_type_detector.detect_type(file_path)

            if not self.can_extract(file_path, document_type):
                raise ExtractionError(
                    f"Docling cannot extract from {document_type.value} files",
                    file_path,
                    ExtractionStrategy.DOCLING,
                )

            # Configurar el converter según las opciones

            # Realizar la conversión
            logger.info(f"Starting Docling extraction for: {file_path}")
            result = self.converter.convert(str(file_path))

            if not result or not hasattr(result, "document"):
                raise ExtractionError(
                    "Docling failed to convert document",
                    file_path,
                    ExtractionStrategy.DOCLING,
                )

            # Extraer contenido
            document = result.document
            text_content = self._extract_text(document, config)
            metadata = self._extract_metadata(document, file_path, document_type)

            # Extraer elementos adicionales si se solicita
            images = self._extract_images(document) if config.extract_images else []
            tables = self._extract_tables(document) if config.extract_tables else []
            structured_content = self._extract_structure(document)

            extraction_time = time.time() - start_time
            logger.info(f"Docling extraction completed in {extraction_time:.2f}s")

            return ExtractedContent(
                text=text_content,
                metadata=metadata,
                images=images,
                tables=tables,
                structure=structured_content,
                extraction_time=extraction_time,
                extraction_strategy=ExtractionStrategy.DOCLING,
                status=ExtractionStatus.SUCCESS,
                source_file=str(file_path),
            )

        except Exception as e:
            extraction_time = time.time() - start_time
            logger.error(f"Docling extraction failed: {str(e)}")

            # Crear metadata básico para el error
            metadata = self._create_metadata(file_path, document_type)

            return ExtractedContent(
                text="",
                metadata=metadata,
                extraction_time=extraction_time,
                extraction_strategy=ExtractionStrategy.DOCLING,
                status=ExtractionStatus.FAILED,
                error_message=str(e),
                source_file=str(file_path),
            )

    def _prepare_converter_config(self, config: ExtractionConfig) -> DoclingConfig:
        """Prepara la configuración para el converter de Docling."""
        # Si hay parámetros personalizados con configuración de Docling
        if config.custom_params and config.custom_params.docling_config:
            return config.custom_params.docling_config

        # Por defecto, crear configuración basada en el modo de extracción
        if config.priority == ProcessingPriority.HIGH:
            return DoclingConfig.create_fast_mode()
        elif config.priority == ProcessingPriority.CRITICAL:
            return DoclingConfig.create_high_accuracy_mode()
        else:
            return DoclingConfig.create_default()

    def _extract_text(self, document, config: ExtractionConfig) -> str:
        """Extrae el texto del documento procesado por Docling."""
        try:
            # Use the enum value properly
            output_format = config.output_format.value.lower()

            if output_format == "markdown":
                text = document.export_to_markdown()
            elif output_format == "html":
                # Si Docling soporta HTML, usar eso, sino convertir markdown
                text = getattr(
                    document, "export_to_html", lambda: document.export_to_markdown()
                )()
            else:
                # Formato texto plano
                text = (
                    document.export_to_text()
                    if hasattr(document, "export_to_text")
                    else str(document)
                )

            # Limpiar el texto si no se quiere preservar formato
            if not config.preserve_formatting:
                text = ContentCleaner.clean_text(text)

            return text

        except Exception as e:
            logger.warning(f"Error extracting text with Docling: {e}")
            return str(document) if document else ""

    def _extract_metadata(
        self, document, file_path: Path, document_type: DocumentType
    ) -> DocumentMetadata:
        """Extrae metadatos del documento procesado por Docling."""
        # Crear metadatos base
        metadata = self._create_metadata(file_path, document_type)

        try:
            # Intentar extraer metadatos específicos de Docling
            if hasattr(document, "metadata"):
                doc_metadata = document.metadata

                if hasattr(doc_metadata, "title"):
                    metadata.title = doc_metadata.title
                if hasattr(doc_metadata, "author"):
                    metadata.author = doc_metadata.author
                if hasattr(doc_metadata, "creator"):
                    metadata.creator = doc_metadata.creator
                if hasattr(doc_metadata, "subject"):
                    metadata.subject = doc_metadata.subject
                if hasattr(doc_metadata, "keywords"):
                    metadata.keywords = doc_metadata.keywords or []

            # Contar páginas si es posible
            if hasattr(document, "pages"):
                metadata.page_count = len(document.pages)
            elif hasattr(document, "page_count"):
                metadata.page_count = document.page_count

            # Contar palabras del texto extraído
            if hasattr(document, "export_to_text"):
                text = document.export_to_text()
                metadata.word_count = len(text.split()) if text else 0

        except Exception as e:
            logger.warning(f"Error extracting metadata with Docling: {e}")

        return metadata

    def _extract_images(self, document) -> list[ImageInfo]:
        """Extrae información de imágenes del documento."""
        images = []
        try:
            if hasattr(document, "pictures") or hasattr(document, "images"):
                doc_images = getattr(
                    document, "pictures", getattr(document, "images", [])
                )

                for idx, img in enumerate(doc_images):
                    image_info = ImageInfo(
                        index=idx,
                        image_type="image",
                        width=getattr(img, "width", None),
                        height=getattr(img, "height", None),
                        format_type=getattr(img, "format", None),
                        data_available=bool(getattr(img, "data", None)),
                    )
                    images.append(image_info)

        except Exception as e:
            logger.warning(f"Error extracting images with Docling: {e}")

        return images

    def _extract_tables(self, document) -> list[TableInfo]:
        """Extrae información de tablas del documento."""
        tables = []
        try:
            if hasattr(document, "tables"):
                for idx, table in enumerate(document.tables):
                    table_info = TableInfo(
                        index=idx,
                        table_type="table",
                        rows=getattr(table, "row_count", 0),
                        columns=getattr(table, "col_count", 0),
                        has_data=bool(
                            hasattr(table, "to_dict") or hasattr(table, "data")
                        ),
                        data_preview=str(
                            getattr(
                                table, "to_dict", lambda t=table: getattr(t, "data", "")
                            )()
                        )[:100],  # Preview first 100 chars
                    )
                    tables.append(table_info)

        except Exception as e:
            logger.warning(f"Error extracting tables with Docling: {e}")

        return tables

    def _extract_structure(self, document) -> DocumentStructure:
        """Extrae información estructural del documento."""
        structure = DocumentStructure()
        try:
            if hasattr(document, "body"):
                body = document.body
                if hasattr(body, "elements"):
                    for element in body.elements:
                        element_type = getattr(element, "element_type", "unknown")
                        text_content = str(element) if element else ""
                        page_number = getattr(element, "page", 0)
                        # Assuming position and level are not directly available from docling element
                        # and can be set to default or derived if needed.
                        structural_element = StructuralElement(
                            element_type=element_type,
                            text_content=text_content,
                            page_number=page_number,
                        )
                        structure.elements.append(structural_element)

            # Information about page count can be added to DocumentStructure if needed
            if hasattr(document, "pages"):
                structure.page_count = len(document.pages)

        except Exception as e:
            logger.warning(f"Error extracting structure with Docling: {e}")

        return structure
