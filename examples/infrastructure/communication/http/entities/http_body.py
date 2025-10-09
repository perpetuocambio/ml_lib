"""HttpBody - Typed representation of HTTP request body."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from infrastructure.communication.http.entities.serializable_data import (
    SerializableData,
)


@dataclass
class HttpBody(ABC):
    """Base class for HTTP request bodies."""

    @abstractmethod
    def to_serializable(self) -> SerializableData:
        """Convert to a serializable object.

        This should be overridden by subclasses.

        Returns:
            Serializable representation
        """
        raise NotImplementedError("Subclasses must implement to_serializable")
