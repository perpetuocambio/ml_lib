"""Download result entities."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path


class DownloadStatus(Enum):
    """Status of a download operation."""
    SUCCESS = "success"
    FAILED = "failed"
    CACHED = "cached"
    IN_PROGRESS = "in_progress"


@dataclass
class DownloadResult:
    """Result of a model download operation."""

    status: DownloadStatus
    model_id: str
    local_path: Path | None = None

    # Download stats
    download_time_seconds: float = 0.0
    downloaded_bytes: int = 0

    # Verification
    checksum_verified: bool = False
    expected_sha256: str = ""
    actual_sha256: str = ""

    # Error info
    error_message: str = ""

    # Metadata
    timestamp: datetime = None

    def __post_init__(self):
        """Initialize timestamp."""
        if self.timestamp is None:
            self.timestamp = datetime.now()

        if isinstance(self.local_path, str):
            self.local_path = Path(self.local_path)

    @property
    def success(self) -> bool:
        """Check if download was successful."""
        return self.status in (DownloadStatus.SUCCESS, DownloadStatus.CACHED)

    @property
    def download_mb(self) -> float:
        """Downloaded size in MB."""
        return self.downloaded_bytes / (1024 * 1024)

    @property
    def download_speed_mbps(self) -> float:
        """Download speed in MB/s."""
        if self.download_time_seconds > 0:
            return self.download_mb / self.download_time_seconds
        return 0.0
