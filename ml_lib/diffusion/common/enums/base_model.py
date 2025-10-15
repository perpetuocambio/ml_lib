from enum import Enum


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
