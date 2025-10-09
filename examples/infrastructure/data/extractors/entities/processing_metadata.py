"""Infrastructure processing metadata DTO - no Domain dependencies."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class InfraProcessingMetadata:
    """Infrastructure DTO for processing metadata."""

    processing_id: str
    document_id: str
    stage: str  # Infrastructure uses string instead of enum
    started_at: datetime
    completed_at: datetime | None
    processing_duration_seconds: float | None

    def is_completed(self) -> bool:
        """Check if processing is completed."""
        return self.completed_at is not None

    def get_processing_duration(self) -> float:
        """Get processing duration in seconds."""
        if self.processing_duration_seconds is not None:
            return self.processing_duration_seconds
        if self.completed_at is not None:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    def get_status_summary(self) -> str:
        """Get a summary of processing status."""
        if self.is_completed():
            return f"Completed {self.stage} in {self.get_processing_duration():.1f}s"
        return f"In progress: {self.stage}"
