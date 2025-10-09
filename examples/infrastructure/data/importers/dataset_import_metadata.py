"""Dataset import metadata structure."""

from dataclasses import dataclass


@dataclass
class DatasetImportMetadata:
    """Metadata collected during dataset import operations."""

    source_format: str = ""
    file_size_bytes: int = 0
    encoding: str = "utf-8"
    delimiter: str = ","
    has_headers: bool = True
    import_method: str = ""
    validation_errors: list[str] = None

    def __post_init__(self):
        """Initialize optional fields if None."""
        if self.validation_errors is None:
            self.validation_errors = []
