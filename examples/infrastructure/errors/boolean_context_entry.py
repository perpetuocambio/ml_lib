"""Boolean context entry for errors."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BooleanContextEntry:
    """Boolean context entry for errors."""

    key: str
    value: bool

    def matches_key(self, key: str) -> bool:
        """Check if this entry matches the given key."""
        return self.key == key

    def get_display_value(self) -> str:
        """Get formatted display value."""
        return f"{self.key}={self.value}"

    def is_true(self) -> bool:
        """Check if the boolean value is True."""
        return self.value is True

    def is_false(self) -> bool:
        """Check if the boolean value is False."""
        return self.value is False

    def get_text_representation(self) -> str:
        """Get text representation of the boolean value."""
        return "yes" if self.value else "no"

    def get_status_representation(self) -> str:
        """Get status representation of the boolean value."""
        return "enabled" if self.value else "disabled"
