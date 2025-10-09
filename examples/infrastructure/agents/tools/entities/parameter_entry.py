"""Parameter entry for infrastructure layer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterEntry:
    """Infrastructure parameter entry."""

    key: str
    value: str

    def __post_init__(self) -> None:
        """Validate parameter entry."""
        if not isinstance(self.key, str) or not isinstance(self.value, str):
            raise ValueError("Key and value must be strings")
        if not self.key.strip():
            raise ValueError("Key cannot be empty")
