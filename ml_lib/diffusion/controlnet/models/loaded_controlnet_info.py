from dataclasses import dataclass
from typing import Optional
from ml_lib.diffusion.controlnet.models.controlnet_config import ControlNetConfig


@dataclass
class LoadedControlNetInfo:
    """Information about a loaded ControlNet model."""

    config: ControlNetConfig
    model: Optional[object]  # Would be ControlNetModel in production
    device: str
