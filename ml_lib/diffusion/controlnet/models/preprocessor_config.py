from dataclasses import dataclass
from ml_lib.diffusion.controlnet.enums.control_type import ControlType


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
