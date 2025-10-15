from enum import Enum


class ModelFormat(Enum):
    """Model file format."""

    SAFETENSORS = "safetensors"
    CKPT = "ckpt"
    DIFFUSERS = "diffusers"
    PICKLE = "pickle"
