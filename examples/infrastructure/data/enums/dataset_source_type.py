"""Dataset source types for import operations."""

from enum import Enum


class DatasetSourceType(Enum):
    """Types of dataset sources for import."""

    FILE_UPLOAD = "FILE_UPLOAD"
    URL_DOWNLOAD = "URL_DOWNLOAD"
    API_ENDPOINT = "API_ENDPOINT"
    DATABASE_QUERY = "DATABASE_QUERY"
