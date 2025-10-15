from enum import Enum


class AdapterType(Enum):
    """Type of adapter."""

    CONTROLNET = "controlnet"
    IPADAPTER = "ipadapter"
    LORA = "lora"
