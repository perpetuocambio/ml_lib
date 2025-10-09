"""Numeric context entry for errors."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NumericContextEntry:
    """Numeric context entry for errors."""

    key: str
    value: int | float

    def matches_key(self, key: str) -> bool:
        """Check if this entry matches the given key."""
        return self.key == key

    def get_display_value(self) -> str:
        """Get formatted display value."""
        return f"{self.key}={self.value}"

    def is_integer(self) -> bool:
        """Check if the numeric value is an integer."""
        return isinstance(self.value, int)

    def is_float(self) -> bool:
        """Check if the numeric value is a float."""
        return isinstance(self.value, float)

    def get_formatted_value(self, decimal_places: int = 2) -> str:
        """Get formatted numeric value with specified decimal places."""
        if self.is_integer():
            return str(self.value)
        return f"{self.value:.{decimal_places}f}"

    def is_positive(self) -> bool:
        """Check if the numeric value is positive."""
        return self.value > 0

    def is_negative(self) -> bool:
        """Check if the numeric value is negative."""
        return self.value < 0

    def is_zero(self) -> bool:
        """Check if the numeric value is zero."""
        return self.value == 0
