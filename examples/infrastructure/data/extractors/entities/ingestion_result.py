"""
Resultado de una operación de ingestión de documentos.
"""

from dataclasses import dataclass

from infrastructure.data.extractors.entities.extracted_content import (
    ExtractedContent,
)


@dataclass
class IngestionResult:
    """Resultado de una operación de ingestión."""

    total_files: int
    successful_extractions: int
    failed_extractions: int
    results: list[ExtractedContent]
    errors: list[str]

    @property
    def success_rate(self) -> float:
        """Tasa de éxito de la ingestión."""
        return (
            (self.successful_extractions / self.total_files * 100)
            if self.total_files > 0
            else 0.0
        )

    def get_summary(self) -> str:
        """Obtiene un resumen de los resultados."""
        return (
            f"Procesados: {self.total_files} archivos, "
            f"Exitosos: {self.successful_extractions}, "
            f"Fallidos: {self.failed_extractions}, "
            f"Tasa de éxito: {self.success_rate:.1f}%"
        )
