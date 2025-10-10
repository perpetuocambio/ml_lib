"""ControlNet entities."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class ControlType(Enum):
    """Type of control signal."""

    CANNY = "canny"
    DEPTH = "depth"
    POSE = "openpose"
    SEGMENTATION = "seg"
    NORMAL = "normal"
    SCRIBBLE = "scribble"
    MLSD = "mlsd"  # Line detection
    HED = "hed"  # Edge detection


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


@dataclass
class ControlImage:
    """Processed control image."""

    control_type: ControlType
    image: np.ndarray
    """Control image as numpy array (H, W, C)."""

    preprocessing_params: dict
    """Parameters used in preprocessing."""

    scale: float = 1.0
    """Strength of this control."""


@dataclass
class PreprocessorConfig:
    """Configuration for image preprocessor."""

    control_type: ControlType
    detect_resolution: int = 512
    image_resolution: int = 512

    # Canny-specific
    low_threshold: int = 100
    high_threshold: int = 200

    # Depth-specific
    depth_model: str = "midas"  # or "zoe"

    # Pose-specific
    include_hand: bool = False
    include_face: bool = False


__all__ = [
    "ControlType",
    "ControlNetConfig",
    "ControlImage",
    "PreprocessorConfig",
]
