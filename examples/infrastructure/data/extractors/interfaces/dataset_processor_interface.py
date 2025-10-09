"""Infrastructure dataset processor interface - no Domain dependencies."""

from abc import ABC, abstractmethod
from pathlib import Path

from infrastructure.data.extractors.entities.data_processing_result import (
    DataProcessingResult,
)
from infrastructure.data.extractors.entities.processing_configuration import (
    ProcessingConfiguration,
)


class IInfraDatasetProcessor(ABC):
    """Infrastructure interface for dataset processing operations."""

    @abstractmethod
    def clean_dataset(
        self,
        dataset_id: str,
        source_file_path: Path,
        output_directory: Path,
        methods: list[str],  # Infrastructure uses strings instead of enums
        configuration: ProcessingConfiguration | None = None,
    ) -> DataProcessingResult:
        """Clean a dataset using specified methods."""
        pass
