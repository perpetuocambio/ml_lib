"""SerializableData - Structured representation of serializable data."""

from dataclasses import dataclass


@dataclass
class SerializableData:
    """Structured representation of serializable data."""

    content: str
