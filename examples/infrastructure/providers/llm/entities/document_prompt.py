"""
Prompt específico para extracción de documentos con soporte multimodal.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentPrompt:
    """
    Prompt simplificado para extracción de documentos.

    Diseñado específicamente para integración con extractores de documentos
    que requieren soporte multimodal (texto + imágenes).
    """

    content: str
    images: list[Path | str] | None = None
    temperature: float = 0.3  # Low creativity for document extraction
    context_window_size: int = 4000  # Sufficient for image descriptions

    def has_images(self) -> bool:
        """Verifica si el prompt incluye imágenes."""
        return bool(self.images)

    def get_image_count(self) -> int:
        """Obtiene el número de imágenes."""
        return len(self.images) if self.images else 0
