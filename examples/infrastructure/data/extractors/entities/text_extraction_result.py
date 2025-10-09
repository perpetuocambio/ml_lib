"""Infrastructure text extraction result DTO - no Domain dependencies."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class InfraTextExtractionResult:
    """Infrastructure DTO for text extraction results."""

    document_id: str
    source_file_path: Path
    extracted_text: str
    extraction_method: str
    confidence_score: float  # Infrastructure uses float instead of ConfidenceMetric
    extraction_timestamp: datetime

    def get_text_length(self) -> int:
        """Get the length of extracted text."""
        return len(self.extracted_text)

    def has_meaningful_content(self) -> bool:
        """Check if extracted text has meaningful content."""
        return len(self.extracted_text.strip()) > 10

    def get_extraction_summary(self) -> str:
        """Get a summary of the extraction result."""
        return (
            f"Extracted {self.get_text_length()} characters "
            f"from {self.source_file_path.name} "
            f"using {self.extraction_method} "
            f"(confidence: {self.confidence_score})"
        )
