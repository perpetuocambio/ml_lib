"""Model metadata entities."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


@dataclass
class ModelMetadata:
    """Complete metadata for a model."""

    # Identity
    model_id: str
    name: str
    source: Source
    type: ModelType
    base_model: BaseModel
    version: str = "main"

    # Technical info
    format: ModelFormat = ModelFormat.SAFETENSORS
    size_bytes: int = 0
    sha256: str = ""

    # Semantic metadata
    trigger_words: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    description: str = ""

    # Stats and recommendations
    download_count: int = 0
    rating: float = 0.0
    recommended_weight: float | None = None

    # Paths
    local_path: Path | None = None
    remote_url: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate and normalize data."""
        # Ensure enums
        if isinstance(self.source, str):
            self.source = Source(self.source)
        if isinstance(self.type, str):
            self.type = ModelType(self.type)
        if isinstance(self.base_model, str):
            self.base_model = BaseModel(self.base_model)
        if isinstance(self.format, str):
            self.format = ModelFormat(self.format)

        # Ensure Path
        if self.local_path and isinstance(self.local_path, str):
            self.local_path = Path(self.local_path)

    @property
    def is_downloaded(self) -> bool:
        """Check if model is downloaded locally."""
        return self.local_path is not None and self.local_path.exists()

    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    @property
    def size_gb(self) -> float:
        """Size in gigabytes."""
        return self.size_bytes / (1024 * 1024 * 1024)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "source": self.source.value,
            "type": self.type.value,
            "base_model": self.base_model.value,
            "version": self.version,
            "format": self.format.value,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "trigger_words": self.trigger_words,
            "tags": self.tags,
            "description": self.description,
            "download_count": self.download_count,
            "rating": self.rating,
            "recommended_weight": self.recommended_weight,
            "local_path": str(self.local_path) if self.local_path else None,
            "remote_url": self.remote_url,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelMetadata":
        """Create from dictionary."""
        # Parse datetimes
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        return cls(**data)


class SortBy(Enum):
    """Sort options for model search."""

    RELEVANCE = "relevance"
    POPULARITY = "popularity"
    DOWNLOADS = "downloads"
    RATING = "rating"
    NEWEST = "newest"
    UPDATED = "updated"


@dataclass
class ModelFilter:
    """Filter criteria for model search."""

    # Type filters
    model_type: ModelType | None = None
    base_model: BaseModel | None = None

    # Content filters
    tags: list[str] = field(default_factory=list)
    min_rating: float = 0.0
    min_downloads: int = 0

    # Technical filters
    max_size_gb: float | None = None
    formats: list[str] = field(default_factory=list)

    # HuggingFace specific
    task: str | None = None  # e.g., "text-to-image"
    library: str | None = None  # e.g., "diffusers"

    # Sorting
    sort_by: SortBy = SortBy.RELEVANCE

    def to_hf_params(self) -> dict:
        """Convert to HuggingFace API parameters."""
        params = {}

        if self.task:
            params["task"] = self.task
        if self.library:
            params["library"] = self.library

        # Tags become filter
        if self.tags:
            params["filter"] = ",".join(self.tags)

        # Sort
        if self.sort_by == SortBy.DOWNLOADS:
            params["sort"] = "downloads"
        elif self.sort_by == SortBy.UPDATED:
            params["sort"] = "lastModified"

        return params

    def to_civitai_params(self) -> dict:
        """Convert to CivitAI API parameters."""
        params = {}

        if self.model_type:
            # Map to CivitAI types
            type_map = {
                ModelType.BASE_MODEL: "Checkpoint",
                ModelType.LORA: "LORA",
                ModelType.EMBEDDING: "TextualInversion",
                ModelType.VAE: "VAE",
                ModelType.CONTROLNET: "Controlnet",
            }
            if self.model_type in type_map:
                params["types"] = type_map[self.model_type]

        if self.base_model and self.base_model != BaseModel.UNKNOWN:
            params["baseModels"] = self.base_model.value.upper()

        if self.tags:
            params["tag"] = ",".join(self.tags)

        # Sort
        if self.sort_by == SortBy.POPULARITY:
            params["sort"] = "Highest Rated"
        elif self.sort_by == SortBy.DOWNLOADS:
            params["sort"] = "Most Downloaded"
        elif self.sort_by == SortBy.NEWEST:
            params["sort"] = "Newest"

        return params

    def matches(self, metadata: "ModelMetadata") -> bool:
        """Check if metadata matches this filter."""
        # Type check
        if self.model_type and metadata.type != self.model_type:
            return False

        # Base model check
        if self.base_model and metadata.base_model != self.base_model:
            return False

        # Rating check
        if metadata.rating < self.min_rating:
            return False

        # Downloads check
        if metadata.download_count < self.min_downloads:
            return False

        # Size check
        if self.max_size_gb and metadata.size_gb > self.max_size_gb:
            return False

        # Tags check (any tag must match)
        if self.tags:
            if not any(tag in metadata.tags for tag in self.tags):
                return False

        # Format check
        if self.formats:
            if metadata.format.value not in self.formats:
                return False

        return True


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
