from dataclasses import dataclass

from infrastructure.config.algorithms.extraction_config import ExtractionConfig


@dataclass
class CustomParameters:
    """Parámetros personalizados para extractores."""

    extraction_config: ExtractionConfig
    additional_notes: str = ""

    @classmethod
    def create_default(cls) -> "CustomParameters":
        """Crea parámetros con configuración por defecto."""
        return cls(extraction_config=ExtractionConfig.from_environment())

    @classmethod
    def create_with_config(
        cls, extraction_config: ExtractionConfig
    ) -> "CustomParameters":
        """Crea parámetros con configuración específica."""
        return cls(extraction_config=extraction_config)
