"""ControlNet models."""

from dataclasses import dataclass
from ml_lib.diffusion.controlnet.enums.control_type import ControlType


@dataclass
class ControlNetConfig:
    """Configuration for ControlNet."""

    model_id: str
    control_type: ControlType
    conditioning_scale: float = 1.0
    """Scale of control signal (0.0 to 2.0)."""

    guess_mode: bool = False
    """If True, ControlNet will try to infer structure without prompt."""

    def __post_init__(self):
        """Validate config."""
        if not 0.0 <= self.conditioning_scale <= 2.0:
            raise ValueError("conditioning_scale must be between 0.0 and 2.0")
