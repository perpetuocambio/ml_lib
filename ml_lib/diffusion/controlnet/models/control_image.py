from dataclasses import dataclass

from ml_lib.diffusion.controlnet.enums.control_type import ControlType
import numpy as np


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
