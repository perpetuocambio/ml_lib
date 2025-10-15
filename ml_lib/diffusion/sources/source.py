from enum import Enum


class Source(Enum):
    """Source of the model."""

    HUGGINGFACE = "huggingface"
    CIVITAI = "civitai"
    LOCAL = "local"
