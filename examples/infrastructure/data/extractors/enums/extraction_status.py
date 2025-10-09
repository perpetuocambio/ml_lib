"""
Estados de extracción de documentos.
"""

from enum import Enum


class ExtractionStatus(Enum):
    """Estados posibles de una extracción de documento."""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
