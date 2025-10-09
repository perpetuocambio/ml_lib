"""Query parameters for database operations."""

from dataclasses import dataclass, field


@dataclass
class QueryParameters:
    """Type-safe query parameters."""

    string_params: list[str] = field(default_factory=list)
    numeric_params: list[float] = field(default_factory=list)
    boolean_params: list[bool] = field(default_factory=list)

    def add_string(self, value: str) -> None:
        """Add string parameter."""
        self.string_params.append(value)

    def add_numeric(self, value: float) -> None:
        """Add numeric parameter."""
        self.numeric_params.append(value)

    def add_boolean(self, value: bool) -> None:
        """Add boolean parameter."""
        self.boolean_params.append(value)

    def get_summary(self) -> str:
        """Get parameters summary."""
        parts = []

        if self.string_params:
            parts.append(f"strings: {len(self.string_params)}")

        if self.numeric_params:
            parts.append(f"numbers: {len(self.numeric_params)}")

        if self.boolean_params:
            parts.append(f"booleans: {len(self.boolean_params)}")

        return ", ".join(parts) if parts else "no parameters"
