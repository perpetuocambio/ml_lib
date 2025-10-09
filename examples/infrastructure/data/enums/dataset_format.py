"""Dataset format types for import operations."""

from enum import Enum


class DatasetFormat(Enum):
    """Supported dataset formats for import."""

    CSV = "CSV"
    JSON = "JSON"
    XML = "XML"
    EXCEL = "EXCEL"
    TSV = "TSV"
    PARQUET = "PARQUET"
