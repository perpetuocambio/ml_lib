"""Collection of tool names for domain operations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolNamesCollection:
    """Type-safe collection of tool names."""

    items: list[str]

    def __post_init__(self) -> None:
        """Validate collection contents."""
        if not all(isinstance(item, str) and item.strip() for item in self.items):
            raise ValueError("All items must be non-empty strings")

    @classmethod
    def create(cls, names: list[str]) -> ToolNamesCollection:
        """Create collection from list of tool names."""
        return cls(items=list(names))

    @classmethod
    def empty(cls) -> ToolNamesCollection:
        """Create empty collection."""
        return cls(items=[])

    def is_empty(self) -> bool:
        """Check if collection is empty."""
        return len(self.items) == 0

    def count(self) -> int:
        """Get count of items."""
        return len(self.items)

    def contains(self, tool_name: str) -> bool:
        """Check if collection contains tool name."""
        return tool_name in self.items
