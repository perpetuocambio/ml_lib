"""
Servicio de ingestión de documentos que puede procesar archivos individuales o directorios completos.
Este servicio orquesta el procesamiento usando el DocumentExtractorFactory.
"""

import logging
from collections.abc import Callable
from pathlib import Path

from infrastructure.config.algorithms.extraction_config import ExtractionConfig
from infrastructure.data.extractors.entities.custom_parameters import (
    CustomParameters,
)
from infrastructure.data.extractors.entities.docling_config import DoclingConfig
from infrastructure.data.extractors.entities.document_metadata import (
    DocumentMetadata,
)
from infrastructure.data.extractors.entities.extracted_content import (
    ExtractedContent,
)
from infrastructure.data.extractors.entities.ingestion_result import (
    IngestionResult,
)
from infrastructure.data.extractors.enums.document_type import DocumentType
from infrastructure.data.extractors.enums.extraction_status import (
    ExtractionStatus,
)
from infrastructure.data.extractors.services.factory import (
    DocumentExtractorFactory,
)
from infrastructure.persistence.storage.interfaces.storage_factory_interface import (
    IStorageFactory,
)
from infrastructure.persistence.storage.interfaces.storage_interface import (
    StorageInterface,
)

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    """
    Servicio para ingerir documentos desde archivos individuales o directorios.

    Este servicio se encarga de:
    - Identificar si la entrada es un archivo o directorio
    - Procesar archivos individuales usando el factory
    - Manejar errores y reportar resultados
    - Filtrar archivos por extensiones soportadas
    """

    def __init__(
        self,
        factory: DocumentExtractorFactory | None = None,
        input_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        storage: StorageInterface | None = None,
        storage_factory: IStorageFactory | None = None,
    ):
        """
        Inicializa el servicio de ingestión.

        Args:
            factory: Factory de extractores. Si no se proporciona, se crea uno nuevo.
            input_dir: Directorio por defecto para inputs. Si no se proporciona, se detecta automáticamente.
            output_dir: Directorio por defecto para outputs. Si no se proporciona, se detecta automáticamente.
            storage: Interfaz de storage.
            storage_factory: Factory para crear storage si storage no se proporciona.
        """
        if storage:
            self.storage = storage
        elif storage_factory:
            self.storage = storage_factory.create_default_storage()
        else:
            raise ValueError("Either storage or storage_factory must be provided")

        self.factory = factory or DocumentExtractorFactory(output_dir, self.storage)
        self.input_dir = (
            Path(input_dir) if input_dir else self._find_default_input_dir()
        )
        self.output_dir = self.factory.output_dir
        self.supported_extensions = self._get_supported_extensions()
        self._default_config = self._create_optimized_config()

    def _find_default_input_dir(self) -> Path:
        """Encuentra el directorio de input por defecto."""
        current_dir = Path(__file__).resolve()

        # Look for the main project structure
        for parent in current_dir.parents:
            if (parent / "pyproject.toml").exists():
                # Found pyintelcivil, go up to find intel directory
                intel_dir = parent.parent
                if intel_dir.name == "intel":
                    return intel_dir / "input"

        # Fallback: look for intel directory in parents
        for parent in current_dir.parents:
            if parent.name == "intel" and (parent / "input").exists():
                return parent / "input"

        # Final fallback to current working directory
        return Path.cwd() / "input"

    def _get_supported_extensions(self) -> set[str]:
        """Obtiene las extensiones de archivo soportadas."""
        # Por ahora definimos manualmente, pero se podría obtener del registry
        return {
            ".pdf",
            ".docx",
            ".doc",
            ".pptx",
            ".ppt",
            ".xlsx",
            ".xls",
            ".html",
            ".htm",
            ".md",
            ".txt",
            ".rtf",
            ".odt",
            ".odp",
            ".ods",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".tiff",
            ".svg",
        }

    def _create_optimized_config(self) -> ExtractionConfig:
        """Crea una configuración optimizada basada en las extensiones soportadas."""
        # Create optimized configuration for docling extractor
        docling_config = DoclingConfig.create_for_file_extensions(
            self.supported_extensions
        )

        custom_params = CustomParameters.create_with_docling(docling_config)

        base_config = ExtractionConfig.create_comprehensive_mode()
        return self._replace_config_with_custom_params(base_config, custom_params)

    def _replace_config_with_custom_params(
        self, base_config: ExtractionConfig, custom_params: CustomParameters
    ) -> ExtractionConfig:
        """Helper para reemplazar custom_params en ExtractionConfig preservando inmutabilidad."""
        return ExtractionConfig(
            strategy=base_config.strategy,
            preserve_formatting=base_config.preserve_formatting,
            extract_images=base_config.extract_images,
            extract_tables=base_config.extract_tables,
            extract_metadata=base_config.extract_metadata,
            output_format=base_config.output_format,
            max_file_size_mb=base_config.max_file_size_mb,
            timeout_seconds=base_config.timeout_seconds,
            priority=base_config.priority,
            custom_params=custom_params,
        )

    def ingest(
        self,
        path: str | Path,
        config: ExtractionConfig | None = None,
        recursive: bool = True,
        file_filter: Callable[[Path], bool] | None = None,
    ) -> IngestionResult:
        """
        Ingiere documentos desde un archivo o directorio.

        Args:
            path: Ruta al archivo o directorio a procesar
            config: Configuración de extracción
            recursive: Si procesar subdirectorios recursivamente
            file_filter: Función para filtrar archivos (opcional)

        Returns:
            IngestionResult: Resultado de la ingestión
        """
        if config is None:
            config = self._default_config

        path = Path(path)

        if not path.exists():
            return IngestionResult(
                total_files=0,
                successful_extractions=0,
                failed_extractions=0,
                results=[],
                errors=[f"Path does not exist: {path}"],
            )

        # Determinar archivos a procesar
        if path.is_file():
            files_to_process = [path]
        elif path.is_dir():
            files_to_process = self._get_files_from_directory(
                path, recursive, file_filter
            )
        else:
            return IngestionResult(
                total_files=0,
                successful_extractions=0,
                failed_extractions=0,
                results=[],
                errors=[f"Path is neither file nor directory: {path}"],
            )

        logger.info(f"Starting ingestion of {len(files_to_process)} files")

        # Procesar archivos
        results = []
        errors = []
        successful_count = 0

        for file_path in files_to_process:
            try:
                logger.debug(f"Processing file: {file_path}")
                result = self.factory.extract_document(file_path, config)
                results.append(result)

                if result.success:
                    successful_count += 1
                    logger.debug(f"Successfully processed: {file_path}")
                else:
                    errors.append(
                        f"Extraction failed for {file_path}: {result.error_message}"
                    )
                    logger.warning(
                        f"Extraction failed for {file_path}: {result.error_message}"
                    )

            except Exception as e:
                error_msg = f"Exception processing {file_path}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg, exc_info=True)

                # Crear un resultado fallido
                failed_result = ExtractedContent(
                    text="",
                    metadata=DocumentMetadata(
                        file_path=file_path,
                        file_size=file_path.stat().st_size if file_path.exists() else 0,
                        document_type=DocumentType.UNKNOWN,
                        title=file_path.name,
                    ),
                    status=ExtractionStatus.FAILED,
                    error_message=str(e),
                    source_file=str(file_path),
                )
                results.append(failed_result)

        ingestion_result = IngestionResult(
            total_files=len(files_to_process),
            successful_extractions=successful_count,
            failed_extractions=len(files_to_process) - successful_count,
            results=results,
            errors=errors,
        )

        logger.info(f"Ingestion completed: {ingestion_result.get_summary()}")
        return ingestion_result

    def _get_files_from_directory(
        self,
        directory: Path,
        recursive: bool,
        file_filter: Callable[[Path], bool] | None,
    ) -> list[Path]:
        """
        Obtiene la lista de archivos a procesar desde un directorio.

        Args:
            directory: Directorio a escanear
            recursive: Si escanear recursivamente
            file_filter: Función opcional para filtrar archivos

        Returns:
            Lista de archivos a procesar
        """
        files = []

        # Patrón de búsqueda
        pattern = "**/*" if recursive else "*"

        for path in directory.glob(pattern):
            if path.is_file() and self._should_process_file(path, file_filter):
                files.append(path)

        # Ordenar por nombre para procesamiento consistente
        files.sort()

        logger.debug(f"Found {len(files)} files to process in {directory}")
        return files

    def _should_process_file(
        self, file_path: Path, file_filter: Callable[[Path], bool] | None
    ) -> bool:
        """
        Determina si un archivo debe ser procesado.

        Args:
            file_path: Ruta del archivo
            file_filter: Función de filtro opcional

        Returns:
            True si el archivo debe procesarse
        """
        # Filtrar archivos ocultos
        if file_path.name.startswith("."):
            return False

        # Verificar extensión soportada
        if file_path.suffix.lower() not in self.supported_extensions:
            logger.debug(f"Skipping unsupported file type: {file_path}")
            return False

        # Aplicar filtro personalizado si existe
        if file_filter and not file_filter(file_path):
            logger.debug(f"File filtered out by custom filter: {file_path}")
            return False

        return True

    def ingest_single_file(
        self, file_path: str | Path, config: ExtractionConfig | None = None
    ) -> ExtractedContent:
        """
        Conveniencia para procesar un solo archivo.

        Args:
            file_path: Ruta del archivo
            config: Configuración de extracción

        Returns:
            ExtractedContent: Resultado de la extracción
        """
        return self.factory.extract_document(file_path, config)

    def get_supported_file_types(self) -> set[str]:
        """Obtiene los tipos de archivo soportados."""
        return self.supported_extensions.copy()
