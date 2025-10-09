"""Dataset import configuration."""

from dataclasses import dataclass

from infrastructure.data.enums.dataset_format import DatasetFormat
from infrastructure.data.enums.dataset_source_type import DatasetSourceType


@dataclass
class DatasetImportConfiguration:
    """Configuration for dataset import operations."""

    source_type: DatasetSourceType
    format: DatasetFormat
    encoding: str = "utf-8"
    delimiter: str | None = None  # For CSV/TSV files
    has_header: bool = True
    max_rows: int | None = None
    skip_rows: int = 0
    chunk_size: int = 10000
    validate_schema: bool = True
    clean_data: bool = True
