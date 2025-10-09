import mimetypes
from pathlib import Path

# Optional dependency - python-magic (declared in pyproject.toml)
# AUTHORIZED: Optional infrastructure dependency with graceful degradation
try:
    import magic

    HAS_MAGIC = True
except ImportError:
    magic = None
    HAS_MAGIC = False

from infrastructure.data.extractors.entities.document_type_mappings import (
    DocumentTypeMappings,
)
from infrastructure.data.extractors.entities.extraction_errors import (
    ExtractionError,
)
from infrastructure.data.extractors.enums.document_type import DocumentType


class DocumentTypeDetector:
    """Detector de tipos de documento basado en extensión y contenido."""

    def __init__(self):
        pass

    def detect_type(self, file_path: Path) -> DocumentType:
        """
        Detecta el tipo de documento basado en la extensión y el contenido.

        Args:
            file_path: Ruta al archivo

        Returns:
            DocumentType: Tipo de documento detectado
        """
        if not file_path.exists():
            raise ExtractionError(f"File not found: {file_path}", file_path)

        # Primero, intentar por extensión
        extension = file_path.suffix
        detected_by_extension = DocumentTypeMappings.get_type_by_extension(extension)

        if detected_by_extension:
            # Verificar con tipo MIME si está disponible
            mime_type = self._get_mime_type(file_path)
            if mime_type:
                detected_by_mime = DocumentTypeMappings.get_type_by_mime_type(mime_type)
                if detected_by_mime:
                    # Si ambos coinciden, usar el detectado
                    if detected_by_extension == detected_by_mime:
                        return detected_by_extension
                    # Si no coinciden, usar el tipo MIME (más confiable)
                    return detected_by_mime

            return detected_by_extension

        # Si no se puede detectar por extensión, intentar por tipo MIME
        mime_type = self._get_mime_type(file_path)
        if mime_type:
            detected_by_mime = DocumentTypeMappings.get_type_by_mime_type(mime_type)
            if detected_by_mime:
                return detected_by_mime

        return DocumentType.UNKNOWN

    def _get_mime_type(self, file_path: Path) -> str | None:
        """
        Obtiene el tipo MIME del archivo.

        Args:
            file_path: Ruta al archivo

        Returns:
            str: Tipo MIME del archivo o None si no se puede determinar
        """
        if HAS_MAGIC and magic is not None:
            try:
                return magic.from_file(str(file_path), mime=True)
            except Exception:
                # Si falla magic por cualquier razón, usar fallback
                pass

        # Fallback a mimetypes si magic no está disponible o falla
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type
