"""Imported dataset representation."""

from dataclasses import dataclass
from datetime import datetime

from infrastructure.data.enums.dataset_format import DatasetFormat
from infrastructure.data.enums.dataset_source_type import DatasetSourceType


@dataclass
class ImportedDataset:
    """Represents an imported dataset."""

    success: bool
    dataset_id: str
    project_id: str
    source_description: str
    source_type: DatasetSourceType
    format: DatasetFormat
    file_path: str
    row_count: int
    column_count: int
    column_names: list[str]
    file_size_bytes: int
    import_timestamp: datetime
    processing_time_seconds: float
    sample_data: list[dict] | None = None
    error_message: str | None = None
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
