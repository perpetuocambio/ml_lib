import logging
from pathlib import Path

from infrastructure.data.extractors.entities.extractor_registration import (
    ExtractorRegistration,
)
from infrastructure.data.extractors.enums.document_type import DocumentType
from infrastructure.data.extractors.enums.extraction_strategy import (
    ExtractionStrategy,
)
from infrastructure.data.extractors.interfaces.base_document_extractor import (
    BaseDocumentExtractor,
)

logger = logging.getLogger(__name__)


class ExtractorRegistry:
    """Registro de extractores disponibles."""

    def __init__(self):
        self._extractors: list[ExtractorRegistration] = []
        self._fallback_order = [
            ExtractionStrategy.DOCLING,
            ExtractionStrategy.MARKITDOWN,
        ]

    def register_extractor(
        self, strategy: ExtractionStrategy, extractor: BaseDocumentExtractor
    ) -> None:
        """Registra un extractor para una estrategia específica."""
        self._extractors.append(
            ExtractorRegistration(strategy=strategy, extractor=extractor)
        )
        logger.info(
            f"Registered extractor: {extractor.name} for strategy: {strategy.value}"
        )

    def get_extractor(
        self, strategy: ExtractionStrategy
    ) -> BaseDocumentExtractor | None:
        """Obtiene un extractor por estrategia."""
        for reg in self._extractors:
            if reg.strategy == strategy:
                return reg.extractor
        return None

    def get_available_strategies(self) -> list[ExtractionStrategy]:
        """Obtiene las estrategias disponibles."""
        return [reg.strategy for reg in self._extractors]

    def get_best_extractor_for_type(
        self, document_type: DocumentType
    ) -> BaseDocumentExtractor | None:
        """Obtiene el mejor extractor disponible para un tipo de documento."""
        for strategy in self._fallback_order:
            for reg in self._extractors:
                if reg.strategy == strategy and reg.extractor.can_extract(
                    Path(), document_type
                ):
                    return reg.extractor
        return None

    def set_fallback_order(self, order: list[ExtractionStrategy]) -> None:
        """Establece el orden de fallback para la selección automática."""
        self._fallback_order = order
