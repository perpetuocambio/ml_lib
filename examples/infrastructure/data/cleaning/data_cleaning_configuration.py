"""Data cleaning configuration for infrastructure layer."""

from dataclasses import dataclass


@dataclass
class DataCleaningConfiguration:
    """Type-safe configuration for data cleaning operations."""

    method_name: str
    threshold_value: float = 0.0
    column_name: str = ""
    pattern: str = ""
    replacement_value: str = ""

    def get_summary(self) -> str:
        """Get configuration summary."""
        parts = [f"method={self.method_name}"]

        if self.threshold_value != 0.0:
            parts.append(f"threshold={self.threshold_value}")

        if self.column_name:
            parts.append(f"column={self.column_name}")

        if self.pattern:
            parts.append(f"pattern={self.pattern}")

        if self.replacement_value:
            parts.append(f"replacement={self.replacement_value}")

        return ", ".join(parts)
