from enum import Enum


class SafetyStatus(Enum):
    """Safety check status."""

    SAFE = "safe"
    UNSAFE = "unsafe"
    WARNING = "warning"
