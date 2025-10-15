from enum import Enum


class DownloadStatus(Enum):
    """Status of a download operation."""

    SUCCESS = "success"
    FAILED = "failed"
    CACHED = "cached"
    IN_PROGRESS = "in_progress"
