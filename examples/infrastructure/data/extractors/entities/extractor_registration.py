from dataclasses import dataclass

from infrastructure.data.extractors.enums.extraction_strategy import (
    ExtractionStrategy,
)
from infrastructure.data.extractors.interfaces.base_document_extractor import (
    BaseDocumentExtractor,
)


@dataclass
class ExtractorRegistration:
    """Registration mapping extraction strategies to their corresponding extractors."""

    strategy: ExtractionStrategy
    extractor: BaseDocumentExtractor
