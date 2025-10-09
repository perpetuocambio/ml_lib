from enum import Enum


class ExtractionStrategy(Enum):
    """Estrategias de extracción disponibles."""

    DOCLING = "docling"
    MARKITDOWN = "markitdown"
    AUTO = "auto"  # Selecciona automáticamente la mejor estrategia
