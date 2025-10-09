"""Model metadata entities."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class Source(Enum):
    """Source of the model."""
    HUGGINGFACE = "huggingface"
    CIVITAI = "civitai"
    LOCAL = "local"


class ModelType(Enum):
    """Type of model."""
    BASE_MODEL = "base_model"
    LORA = "lora"
    EMBEDDING = "embedding"
    VAE = "vae"
    CONTROLNET = "controlnet"
    IPADAPTER = "ipadapter"


class BaseModel(Enum):
    """Base model architecture."""
    SD15 = "sd15"
    SD20 = "sd20"
    SD21 = "sd21"
    SDXL = "sdxl"
    SD3 = "sd3"
    PONY = "pony"
    FLUX = "flux"
    UNKNOWN = "unknown"


class ModelFormat(Enum):
    """Model file format."""
    SAFETENSORS = "safetensors"
    CKPT = "ckpt"
    DIFFUSERS = "diffusers"
    PICKLE = "pickle"


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
