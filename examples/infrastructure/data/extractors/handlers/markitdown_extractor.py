import logging
import re
import time
from pathlib import Path

try:
    import markdown
except ImportError:
    markdown = None

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
from infrastructure.data.extractors.entities.extraction_errors import (
    ExtractionError,
)
from infrastructure.data.extractors.entities.image_info import ImageInfo
from infrastructure.data.extractors.entities.table_info import TableInfo
from infrastructure.data.extractors.enums.document_type import DocumentType
from infrastructure.data.extractors.enums.extraction_status import (
    ExtractionStatus,
)
from infrastructure.data.extractors.enums.extraction_strategy import (
    ExtractionStrategy,
)
from infrastructure.data.extractors.handlers.content_cleaner import ContentCleaner
from infrastructure.data.extractors.handlers.document_type_detector import (
    DocumentTypeDetector,
)
from infrastructure.data.extractors.interfaces.base_document_extractor import (
    BaseDocumentExtractor,
)
from infrastructure.serialization.protocol_serializer import ProtocolSerializer

try:
    from infrastructure.data.extractors.adapters.llm_markitdown_adapter import (
        LLMMarkItDownAdapter,
    )
    from infrastructure.providers.llm.factories.ollama_factory import (
        OllamaFactory,
    )
except ImportError:
    LLMMarkItDownAdapter = None
    OllamaFactory = None

from markitdown import MarkItDown

logger = logging.getLogger(__name__)


class MarkItDownExtractor(BaseDocumentExtractor):
    """Extractor de documentos usando la librería MarkItDown."""

    # MarkItDown es muy flexible y soporta muchos formatos
    SUPPORTED_TYPES = [
        DocumentType.PDF,
        DocumentType.DOCX,
        DocumentType.DOC,
        DocumentType.PPTX,
        DocumentType.PPT,
        DocumentType.XLSX,
        DocumentType.XLS,
        DocumentType.HTML,
        DocumentType.TXT,
        DocumentType.MD,
        DocumentType.RTF,
        DocumentType.ODT,
        DocumentType.ODP,
        DocumentType.ODS,
    ]

    def __init__(self, document_type_detector: DocumentTypeDetector):
        super().__init__("MarkItDownExtractor")
        self._markitdown = None
        self.document_type_detector = document_type_detector
        self.content_cleaner = ContentCleaner()

    @property
    def markitdown(self) -> MarkItDown:
        """Lazy loading del objeto MarkItDown."""
        if self._markitdown is None:
            self._markitdown = MarkItDown()
        return self._markitdown

    def can_extract(self, file_path: Path, document_type: DocumentType) -> bool:
        """Verifica si MarkItDown puede procesar el tipo de documento."""
        return document_type in self.SUPPORTED_TYPES

    def get_supported_types(self) -> list[DocumentType]:
        """Retorna los tipos de documento soportados por MarkItDown."""
        return self.SUPPORTED_TYPES.copy()

    def get_capabilities(self) -> ExtractionCapabilities:
        """Retorna las capacidades del extractor MarkItDown."""
        return ExtractionCapabilities(
            supported_types=self.get_supported_types(),
            can_extract_images=True,
            can_extract_tables=True,
            can_extract_metadata=True,
            can_preserve_formatting=True,  # MarkItDown preserves formatting by default
            supports_ocr=False,  # MarkItDown does not do OCR
            max_file_size_mb=100,  # Default max file size
        )

    def extract(self, file_path: Path, config: ExtractionConfig) -> ExtractedContent:
        """
        Extrae contenido usando MarkItDown.

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
                    f"MarkItDown cannot extract from {document_type.value} files",
                    file_path,
                    ExtractionStrategy.MARKITDOWN,
                )

            # Realizar la conversión
            logger.info(f"Starting MarkItDown extraction for: {file_path}")

            # Configurar MarkItDown según las opciones
            markitdown_instance = self._prepare_markitdown(config)

            # Convertir el documento
            result = markitdown_instance.convert(str(file_path))

            if not result or not hasattr(result, "text_content"):
                raise ExtractionError(
                    "MarkItDown failed to convert document",
                    file_path,
                    ExtractionStrategy.MARKITDOWN,
                )

            # Extraer y procesar contenido
            text_content = self._process_text(result.text_content, config)
            metadata = self._extract_metadata(result, file_path, document_type)

            # MarkItDown generalmente no extrae imágenes o tablas por separado,
            # pero podemos intentar extraer información estructural del markdown
            images = (
                self._extract_images_from_markdown(text_content)
                if config.extract_images
                else []
            )
            tables = (
                self._extract_tables_from_markdown(text_content)
                if config.extract_tables
                else []
            )
            structured_content = self._extract_structure_from_markdown(text_content)

            extraction_time = time.time() - start_time
            logger.info(f"MarkItDown extraction completed in {extraction_time:.2f}s")

            return ExtractedContent(
                text=text_content,
                metadata=metadata,
                images=images,
                tables=tables,
                structure=structured_content,
                extraction_time=extraction_time,
                extraction_strategy=ExtractionStrategy.MARKITDOWN,
                status=ExtractionStatus.SUCCESS,
                source_file=str(file_path),
            )

        except Exception as e:
            extraction_time = time.time() - start_time
            logger.error(f"MarkItDown extraction failed: {str(e)}")

            # Crear metadata básico para el error
            metadata = self._create_metadata(file_path, document_type)

            return ExtractedContent(
                text="",
                metadata=metadata,
                extraction_time=extraction_time,
                extraction_strategy=ExtractionStrategy.MARKITDOWN,
                status=ExtractionStatus.FAILED,
                error_message=str(e),
                source_file=str(file_path),
            )

    def _prepare_markitdown(self, config: ExtractionConfig) -> MarkItDown:
        """Prepara una instancia de MarkItDown con la configuración especificada."""
        markitdown_options = ProtocolSerializer.serialize_markitdown_options({})

        # Use new MarkItDownConfig if available
        if config.custom_params and config.custom_params.markitdown_config:
            markitdown_config = config.custom_params.markitdown_config

            # Configure basic MarkItDown options
            markitdown_options["enable_plugins"] = markitdown_config.enable_plugins

            # Configure LLM client for image descriptions if Ollama integration is enabled
            if markitdown_config.use_ollama_integration and markitdown_config.llm_model:
                try:
                    if LLMMarkItDownAdapter is None or OllamaFactory is None:
                        raise ImportError("Required Ollama dependencies not available")

                    # Create Ollama provider
                    ollama_provider = OllamaFactory.create_vision_provider(
                        markitdown_config.llm_model
                    )

                    # Create adapter for MarkItDown compatibility
                    llm_adapter = LLMMarkItDownAdapter(
                        ollama_provider, markitdown_config.llm_model
                    )

                    # Set up MarkItDown with LLM client
                    markitdown_options["llm_client"] = llm_adapter
                    markitdown_options["llm_model"] = markitdown_config.llm_model

                    if markitdown_config.llm_prompt:
                        markitdown_options["llm_prompt"] = markitdown_config.llm_prompt

                    logger.info(
                        f"Configured MarkItDown with Ollama model: {markitdown_config.llm_model}"
                    )

                except Exception as e:
                    logger.warning(f"Failed to configure Ollama integration: {e}")

            # Add Azure Document Intelligence endpoint if configured
            if markitdown_config.docintel_endpoint:
                markitdown_options["docintel_endpoint"] = (
                    markitdown_config.docintel_endpoint
                )

        # Crear nueva instancia con opciones si es necesario
        if markitdown_options:
            return MarkItDown(**markitdown_options)
        else:
            return self.markitdown

    def _process_text(self, text: str, config: ExtractionConfig) -> str:
        """Procesa el texto extraído según la configuración."""
        if not text:
            return ""

        # Convertir formato si es necesario
        if (
            config.output_format.value.lower() == "text"
            or config.output_format.value.lower() == "plain"
        ):
            # Convertir markdown a texto plano
            text = self._markdown_to_text(text)
        elif config.output_format.value.lower() == "html":
            # Convertir markdown a HTML (requeriría una librería adicional como markdown)
            if markdown is not None:
                text = markdown.markdown(text)
            else:
                logger.warning("markdown library not available, keeping as markdown")
        # Si es markdown, mantener como está

        # Limpiar el texto si no se quiere preservar formato
        if not config.preserve_formatting:
            text = ContentCleaner.clean_text(text)

        return text

    def _markdown_to_text(self, markdown_text: str) -> str:
        """Convierte markdown a texto plano simple."""

        # Remover encabezados markdown
        text = re.sub(r"^#{1,6}\s+", "", markdown_text, flags=re.MULTILINE)

        # Remover formato de texto (negrita, cursiva)
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # negrita
        text = re.sub(r"\*(.*?)\*", r"\1", text)  # cursiva
        text = re.sub(r"__(.*?)__", r"\1", text)  # negrita alternativa
        text = re.sub(r"_(.*?)_", r"\1", text)  # cursiva alternativa

        # Remover enlaces
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Remover código inline
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # Remover bloques de código
        text = re.sub(r"```[\s\S]*?```", "", text)

        # Limpiar listas
        text = re.sub(r"^[\s]*[-*+]\s+", "• ", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

        return text

    def _extract_metadata(
        self, result, file_path: Path, document_type: DocumentType
    ) -> DocumentMetadata:
        """Extrae metadatos del resultado de MarkItDown."""
        # Crear metadatos base
        metadata = self._create_metadata(file_path, document_type)

        try:
            # MarkItDown puede incluir metadatos en el resultado
            if hasattr(result, "title") and result.title:
                metadata.title = result.title

            # Extraer metadatos del contenido si están disponibles
            if hasattr(result, "text_content"):
                text = result.text_content

                # Contar palabras
                metadata.word_count = len(text.split()) if text else 0

                # Intentar extraer título del primer encabezado si no hay título
                if not metadata.title:
                    first_header = re.search(r"^#\s+(.+)", text, re.MULTILINE)
                    if first_header:
                        metadata.title = first_header.group(1).strip()

        except Exception as e:
            logger.warning(f"Error extracting metadata with MarkItDown: {e}")

        return metadata

    def _extract_images_from_markdown(self, markdown_text: str) -> list[ImageInfo]:
        """Extrae referencias de imágenes del texto markdown."""

        images = []

        try:
            # Buscar sintaxis de imágenes en markdown: ![alt](url)
            image_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
            matches = re.finditer(image_pattern, markdown_text)

            for idx, match in enumerate(matches):
                alt_text = match.group(1)
                url = match.group(2)

                image_info = ImageInfo(
                    index=idx,
                    image_type="image_reference",
                    alt_text=alt_text,
                    url=url,
                    position=match.start(),
                )

                images.append(image_info)

        except Exception as e:
            logger.warning(f"Error extracting images from markdown: {e}")

        return images

    def _extract_tables_from_markdown(self, markdown_text: str) -> list[TableInfo]:
        """Extrae tablas del texto markdown."""
        tables = []

        try:
            # Buscar tablas en formato markdown
            # Patrón simple para detectar tablas markdown
            lines = markdown_text.split("\n")
            table_start = None
            current_table = []

            for i, line in enumerate(lines):
                line = line.strip()

                # Detectar líneas de tabla (contienen |)
                if "|" in line and line.count("|") >= 2:
                    if table_start is None:
                        table_start = i
                    current_table.append(line)
                else:
                    # Fin de tabla
                    if table_start is not None and len(current_table) > 0:
                        # Procesar la tabla encontrada
                        table_info = self._parse_markdown_table(
                            current_table, len(tables), table_start
                        )
                        if table_info:
                            tables.append(table_info)

                        # Reset para la siguiente tabla
                        table_start = None
                        current_table = []

            # Procesar última tabla si existe
            if table_start is not None and len(current_table) > 0:
                table_info = self._parse_markdown_table(
                    current_table, len(tables), table_start
                )
                if table_info:
                    tables.append(table_info)

        except Exception as e:
            logger.warning(f"Error extracting tables from markdown: {e}")

        return tables

    def _parse_markdown_table(
        self, table_lines: list[str], table_index: int, start_line: int
    ) -> TableInfo | None:
        """Parsea una tabla markdown individual."""
        try:
            if len(table_lines) < 2:
                return None

            # Primera línea son los headers
            headers = [cell.strip() for cell in table_lines[0].split("|")[1:-1]]

            # Segunda línea debe ser separador (---)
            data_start = 2 if len(table_lines) > 1 and "---" in table_lines[1] else 1

            # Resto son datos
            rows = []
            for line in table_lines[data_start:]:
                cells = [cell.strip() for cell in line.split("|")[1:-1]]
                if len(cells) == len(headers):
                    rows.append(cells)

            return TableInfo(
                index=table_index,
                table_type="markdown_table",
                start_line=start_line + 1,
                headers=headers,
                rows=len(rows),
                columns=len(headers),
                has_data=True,
                data_preview=str({"headers": headers, "rows": rows})[:100],
            )

        except Exception as e:
            logger.warning(f"Error parsing markdown table: {e}")
            return None

    def _extract_structure_from_markdown(self, markdown_text: str) -> DocumentStructure:
        """Extrae información estructural del texto markdown."""
        structure = self.content_cleaner.extract_structure(markdown_text)

        try:
            # Información adicional específica de markdown
            # These are counts, not complex data structures, so keeping them as simple attributes or properties of DocumentStructure
            # is more appropriate than creating new dataclasses for each count.
            # If DocumentStructure needs to store these, they should be added as fields to DocumentStructure.
            # For now, I will remove the dictionary assignment and assume these counts are handled within DocumentStructure or not needed.
            pass

        except Exception as e:
            logger.warning(f"Error extracting markdown structure: {e}")

        return structure
