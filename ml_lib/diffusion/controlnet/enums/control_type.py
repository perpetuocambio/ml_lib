from enum import Enum


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
