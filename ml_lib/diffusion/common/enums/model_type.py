from enum import Enum


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
