"""Error context for infrastructure exceptions."""

from dataclasses import dataclass

from infrastructure.errors.boolean_context_entry import BooleanContextEntry
from infrastructure.errors.numeric_context_entry import NumericContextEntry
from infrastructure.errors.text_context_entry import TextContextEntry


@dataclass(frozen=True)
class ErrorContext:
    """Context information for infrastructure errors."""

    text_entries: list[TextContextEntry]
    numeric_entries: list[NumericContextEntry]
    boolean_entries: list[BooleanContextEntry]

    def get_text_value(self, key: str) -> str | None:
        """Get text value for specific key."""
        for entry in self.text_entries:
            if entry.key == key:
                return entry.value
        return None

    def get_numeric_value(self, key: str) -> int | float | None:
        """Get numeric value for specific key."""
        for entry in self.numeric_entries:
            if entry.key == key:
                return entry.value
        return None

    def get_boolean_value(self, key: str) -> bool | None:
        """Get boolean value for specific key."""
        for entry in self.boolean_entries:
            if entry.key == key:
                return entry.value
        return None

    def has_key(self, key: str) -> bool:
        """Check if key exists in any context type."""
        return (
            any(entry.key == key for entry in self.text_entries)
            or any(entry.key == key for entry in self.numeric_entries)
            or any(entry.key == key for entry in self.boolean_entries)
        )

    def add_text_entry(self, key: str, value: str) -> "ErrorContext":
        """Create new context with additional text entry."""
        new_entries = list(self.text_entries)
        new_entries.append(TextContextEntry(key, value))
        return ErrorContext(new_entries, self.numeric_entries, self.boolean_entries)

    def add_numeric_entry(self, key: str, value: int | float) -> "ErrorContext":
        """Create new context with additional numeric entry."""
        new_entries = list(self.numeric_entries)
        new_entries.append(NumericContextEntry(key, value))
        return ErrorContext(self.text_entries, new_entries, self.boolean_entries)

    def add_boolean_entry(self, key: str, value: bool) -> "ErrorContext":
        """Create new context with additional boolean entry."""
        new_entries = list(self.boolean_entries)
        new_entries.append(BooleanContextEntry(key, value))
        return ErrorContext(self.text_entries, self.numeric_entries, new_entries)

    def get_summary(self) -> str:
        """Get human-readable summary of all context entries."""
        parts = []

        for entry in self.text_entries:
            parts.append(f"{entry.key}={entry.value}")

        for entry in self.numeric_entries:
            parts.append(f"{entry.key}={entry.value}")

        for entry in self.boolean_entries:
            parts.append(f"{entry.key}={entry.value}")

        return ", ".join(parts)

    def is_empty(self) -> bool:
        """Check if context has no entries."""
        return (
            len(self.text_entries) == 0
            and len(self.numeric_entries) == 0
            and len(self.boolean_entries) == 0
        )

    @classmethod
    def empty(cls) -> "ErrorContext":
        """Create empty error context."""
        return cls([], [], [])

    @classmethod
    def with_text(cls, key: str, value: str) -> "ErrorContext":
        """Create context with single text entry."""
        return cls([TextContextEntry(key, value)], [], [])

    @classmethod
    def with_numeric(cls, key: str, value: int | float) -> "ErrorContext":
        """Create context with single numeric entry."""
        return cls([], [NumericContextEntry(key, value)], [])

    @classmethod
    def with_boolean(cls, key: str, value: bool) -> "ErrorContext":
        """Create context with single boolean entry."""
        return cls([], [], [BooleanContextEntry(key, value)])
