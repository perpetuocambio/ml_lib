# utils.py
from pathlib import Path

from infrastructure.data.extractors.entities.extraction_errors import (
    ExtractionError,
)
from infrastructure.persistence.storage.interfaces.storage_interface import (
    StorageInterface,
)


class FileValidator:
    """Validador de archivos para extracción."""

    @staticmethod
    def validate_file(
        file_path: Path, storage: StorageInterface, max_size_mb: int = 100
    ) -> None:
        """
        Valida que el archivo sea accesible y cumpla con los criterios.

        Args:
            file_path: Ruta al archivo
            max_size_mb: Tamaño máximo en MB
            storage: Storage interface implementation

        Raises:
            ExtractionError: Si el archivo no es válido
        """
        file_path_str = str(file_path)

        if not storage.file_exists(file_path_str):
            raise ExtractionError(f"File does not exist: {file_path}", file_path)

        # Get file metadata using storage interface
        try:
            metadata = storage.get_file_metadata(file_path_str)
        except Exception as e:
            raise ExtractionError(f"Cannot access file: {file_path}", file_path) from e

        if metadata.is_directory():
            raise ExtractionError(f"Path is not a file: {file_path}", file_path)

        # Verificar permisos de lectura usando storage interface
        try:
            # Intentar leer el primer byte para verificar permisos
            data = storage.read_file(file_path_str)
            if len(data) == 0:
                # File is empty but readable
                pass
        except PermissionError:
            raise ExtractionError(
                f"File is not readable: {file_path}", file_path
            ) from None
        except Exception as e:
            raise ExtractionError(f"Cannot read file: {file_path}", file_path) from e

        # Verificar tamaño usando metadata
        file_size_mb = metadata.get_size_bytes() / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise ExtractionError(
                f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({max_size_mb} MB)",
                file_path,
            )
