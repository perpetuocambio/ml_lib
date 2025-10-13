"""Model enums - separated to avoid circular imports."""

from enum import Enum


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
    CLIP = "clip"
    CLIP_VISION = "clip_vision"
    TEXT_ENCODER = "text_encoder"
    UNET = "unet"


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


class SortBy(Enum):
    """Sort options for model search."""
    RELEVANCE = "relevance"
    POPULARITY = "popularity"
    DOWNLOADS = "downloads"
    RATING = "rating"
    NEWEST = "newest"
    UPDATED = "updated"


class DownloadStatus(Enum):
    """Status of a download operation."""
    SUCCESS = "success"
    FAILED = "failed"
    CACHED = "cached"
    IN_PROGRESS = "in_progress"
