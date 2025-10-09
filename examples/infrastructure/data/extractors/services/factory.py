# factory.py
import datetime
import logging
from pathlib import Path

from infrastructure.config.algorithms.extraction_config import ExtractionConfig
from infrastructure.data.extractors.entities.extracted_content import (
    ExtractedContent,
)
from infrastructure.data.extractors.entities.extraction_errors import (
    ExtractionError,
)
from infrastructure.data.extractors.entities.extractor_registration import (
    ExtractorRegistration,
)
from infrastructure.data.extractors.enums.document_type import DocumentType
from infrastructure.data.extractors.enums.extraction_strategy import (
    ExtractionStrategy,
)
from infrastructure.data.extractors.handlers.docling_extractor import (
    DoclingExtractor,
)
from infrastructure.data.extractors.handlers.document_type_detector import (
    DocumentTypeDetector,
)
from infrastructure.data.extractors.handlers.exttractor_registry import (
    ExtractorRegistry,
)
from infrastructure.data.extractors.handlers.file_validator import FileValidator
from infrastructure.data.extractors.handlers.markitdown_extractor import (
    MarkItDownExtractor,
)
from infrastructure.data.extractors.interfaces.base_document_extractor import (
    BaseDocumentExtractor,
)
from infrastructure.persistence.storage.interfaces.storage_factory_interface import (
    IStorageFactory,
)
from infrastructure.persistence.storage.interfaces.storage_interface import (
    StorageInterface,
)

logger = logging.getLogger(__name__)


class DocumentExtractorFactory:
    """Factory para crear y gestionar extractores de documentos."""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        storage: StorageInterface | None = None,
        storage_factory: IStorageFactory | None = None,
    ):
        self.registry = ExtractorRegistry()
        self.document_type_detector = DocumentTypeDetector()
        self.output_dir = (
            Path(output_dir) if output_dir else self._find_default_output_dir()
        )

        if storage:
            self.storage = storage
        elif storage_factory:
            self.storage = storage_factory.create_default_storage()
        else:
            raise ValueError("Either storage or storage_factory must be provided")

        self._initialize_extractors()

    def _initialize_extractors(self) -> None:
        """Inicializa los extractores disponibles."""

        self.registry.register_extractor(
            ExtractionStrategy.DOCLING,
            DoclingExtractor(self.document_type_detector),
        )

        self.registry.register_extractor(
            ExtractionStrategy.MARKITDOWN,
            MarkItDownExtractor(self.document_type_detector),
        )

    def _find_default_output_dir(self) -> Path:
        """Encuentra el directorio de output por defecto."""
        current_dir = Path(__file__).resolve()

        # Look for the main project structure
        for parent in current_dir.parents:
            if (parent / "pyproject.toml").exists():
                # Found pyintelcivil, go up to find intel directory
                intel_dir = parent.parent
                if intel_dir.name == "intel":
                    return intel_dir / "output"

        # Fallback: look for intel directory in parents
        for parent in current_dir.parents:
            if parent.name == "intel" and (parent / "input").exists():
                return parent / "output"

        # Final fallback to current working directory
        return Path.cwd() / "output"

    def extract_document(
        self, file_path: str | Path, config: ExtractionConfig | None = None
    ) -> ExtractedContent:
        """
        Extrae contenido de un documento usando la mejor estrategia disponible.

        Args:
            file_path: Ruta al archivo
            config: Configuración de extracción

        Returns:
            ExtractedContent: Contenido extraído
        """
        if config is None:
            config = ExtractionConfig()

        file_path = Path(file_path)

        # Validar archivo usando storage interface
        FileValidator.validate_file(file_path, self.storage, config.max_file_size_mb)

        # Detectar tipo de documento
        document_type = self.document_type_detector.detect_type(file_path)

        if document_type == DocumentType.UNKNOWN:
            raise ExtractionError(
                f"Unknown document type for file: {file_path}", file_path
            )

        # Seleccionar extractor
        extractor = self._select_extractor(document_type, config.strategy)

        if not extractor:
            raise ExtractionError(
                f"No suitable extractor found for {document_type.value} files",
                file_path,
            )

        # Create timestamped output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base_dir = self.output_dir / timestamp
        output_raw_dir = output_base_dir / "raw"
        output_content_dir = output_base_dir / "content"

        output_raw_dir.mkdir(parents=True, exist_ok=True)
        output_content_dir.mkdir(parents=True, exist_ok=True)

        # Copy original file to raw directory using storage interface
        dest_path = str(output_raw_dir / file_path.name)
        self.storage.copy_file(str(file_path), dest_path)

        # Perform extraction
        try:
            result = extractor.extract(file_path, config)
            logger.info(
                f"Successfully extracted content from {file_path} using {extractor.name}"
            )

            # Save extracted text content using storage interface
            if result.text:
                output_text_file = str(
                    output_content_dir / f"{file_path.stem}_extracted_text.md"
                )
                self.storage.write_text_file(output_text_file, result.text, "utf-8")

            # TODO: Save other extracted data (images, tables, metadata)

            return result
        except Exception as e:
            logger.error(f"Extraction failed with {extractor.name}: {e}")

            # Attempt fallback if strategy was AUTO
            if config.strategy == ExtractionStrategy.AUTO:
                return self._try_fallback_extractors(
                    file_path, config, document_type, extractor
                )
            else:
                raise

    def _select_extractor(
        self, document_type: DocumentType, strategy: ExtractionStrategy
    ) -> BaseDocumentExtractor | None:
        """Selecciona el extractor apropiado."""
        if strategy == ExtractionStrategy.AUTO:
            return self.registry.get_best_extractor_for_type(document_type)
        else:
            extractor = self.registry.get_extractor(strategy)
            if extractor and extractor.can_extract(Path(), document_type):
                return extractor
            return None

    def _try_fallback_extractors(
        self,
        file_path: Path,
        config: ExtractionConfig,
        document_type: DocumentType,
        failed_extractor: BaseDocumentExtractor,
    ) -> ExtractedContent:
        """Intenta extractores alternativos como fallback."""
        available_strategies = self.registry.get_available_strategies()

        for strategy in available_strategies:
            extractor = self.registry.get_extractor(strategy)

            # Saltar el extractor que ya falló
            if extractor == failed_extractor:
                continue

            if extractor and extractor.can_extract(file_path, document_type):
                try:
                    logger.info(f"Trying fallback extractor: {extractor.name}")
                    result = extractor.extract(file_path, config)
                    logger.info(f"Fallback extraction successful with {extractor.name}")
                    return result
                except Exception as e:
                    logger.warning(
                        f"Fallback extractor {extractor.name} also failed: {e}"
                    )
                    continue

        # Si todos los extractores fallan
        raise ExtractionError(
            f"All available extractors failed for file: {file_path}", file_path
        )

    def get_supported_types(self) -> list[ExtractorRegistration]:
        """Obtiene los tipos soportados por cada estrategia."""
        return self.registry._extractors

    def register_custom_extractor(
        self, strategy: ExtractionStrategy, extractor: BaseDocumentExtractor
    ) -> None:
        """Registra un extractor personalizado."""
        self.registry.register_extractor(strategy, extractor)

    def set_extraction_priority(self, order: list[ExtractionStrategy]) -> None:
        """Establece el orden de prioridad para extractores automáticos."""
        self.registry.set_fallback_order(order)
