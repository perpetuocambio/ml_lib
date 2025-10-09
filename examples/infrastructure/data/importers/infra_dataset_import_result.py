"""Infrastructure dataset import result DTO."""

from dataclasses import dataclass
from pathlib import Path

from infrastructure.data.importers.dataset_import_metadata import (
    DatasetImportMetadata,
)


@dataclass
class InfraDatasetImportResult:
    """Infrastructure DTO for dataset import results."""

    import_id: str
    dataset_name: str
    file_path: Path | None
    row_count: int
    column_count: int
    columns: list[str]
    success: bool
    import_time_seconds: float
    error_message: str | None = None
    metadata: DatasetImportMetadata = None
