"""Text context entry for errors."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TextContextEntry:
    """Text context entry for errors."""

    key: str
    value: str

    def matches_key(self, key: str) -> bool:
        """Check if this entry matches the given key."""
        return self.key == key

    def get_display_value(self) -> str:
        """Get formatted display value."""
        return f"{self.key}={self.value}"

    def is_empty_value(self) -> bool:
        """Check if the text value is empty or whitespace only."""
        return not self.value or self.value.strip() == ""

    def get_truncated_value(self, max_length: int = 50) -> str:
        """Get truncated value for display purposes."""
        if len(self.value) <= max_length:
            return self.value
        return self.value[:max_length] + "..."
