from pathlib import Path

from infrastructure.data.extractors.enums.extraction_strategy import (
    ExtractionStrategy,
)


class ExtractionError(Exception):
    """Excepción personalizada para errores de extracción."""

    def __init__(
        self,
        message: str,
        file_path: Path | None = None,
        strategy: ExtractionStrategy | None = None,
    ):
        super().__init__(message)
        self.file_path = file_path
        self.strategy = strategy
        self.message = message
