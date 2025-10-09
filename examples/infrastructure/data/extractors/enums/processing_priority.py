"""
Prioridades de procesamiento de documentos.
"""

from enum import Enum


class ProcessingPriority(Enum):
    """Prioridades de procesamiento."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
