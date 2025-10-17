"""Reasoning value objects for decision explanations.

This module provides type-safe reasoning classes WITHOUT using dicts, tuples, or any.
"""

from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class ReasoningEntry:
    """Represents a single reasoning entry.

    Attributes:
        key: The decision key (e.g., "lora_selection", "cfg_value").
        explanation: Human-readable explanation for the decision.

    Example:
        >>> entry = ReasoningEntry("lora_selection", "Selected for photorealism")
        >>> print(entry.key, entry.explanation)
        lora_selection Selected for photorealism
    """

    key: str
    explanation: str

    def __post_init__(self) -> None:
        """Validate reasoning entry."""
        if not self.key:
            raise ValueError("Reasoning key cannot be empty")
        if not self.explanation:
            raise ValueError(f"Reasoning explanation for '{self.key}' cannot be empty")


@dataclass(frozen=True)
class ReasoningMap:
    """Collection of reasoning entries explaining decisions.

    Attributes:
        _entries: Internal list of ReasoningEntry instances.

    Example:
        >>> entries = [
        ...     ReasoningEntry("lora_selection", "Selected for photorealism"),
        ...     ReasoningEntry("cfg_value", "Increased for detail")
        ... ]
        >>> reasoning = ReasoningMap(entries)
        >>> print(reasoning.get("lora_selection"))
        Selected for photorealism
    """

    _entries: list[ReasoningEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate reasoning map."""
        if not self._entries:
            raise ValueError("ReasoningMap cannot be empty")

        # Check for duplicate keys
        keys = [e.key for e in self._entries]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate reasoning keys found")

    @property
    def count(self) -> int:
        """Number of reasoning entries."""
        return len(self._entries)

    @property
    def keys(self) -> list[str]:
        """List of reasoning keys."""
        return [e.key for e in self._entries]

    def get(self, key: str, default: str | None = None) -> str | None:
        """Get explanation for a key.

        Args:
            key: Reasoning key.
            default: Default value if not found.

        Returns:
            Explanation or default value.
        """
        for e in self._entries:
            if e.key == key:
                return e.explanation
        return default

    def has(self, key: str) -> bool:
        """Check if reasoning exists for a key.

        Args:
            key: Reasoning key.

        Returns:
            True if exists, False otherwise.
        """
        return any(e.key == key for e in self._entries)

    def get_all(self) -> list[ReasoningEntry]:
        """Get all reasoning entries.

        Returns:
            List of all ReasoningEntry instances.
        """
        return self._entries.copy()

    def __iter__(self) -> Iterator[ReasoningEntry]:
        """Iterate over reasoning entries."""
        return iter(self._entries)

    def format_for_display(self, indent: str = "  ") -> str:
        """Format reasoning for human-readable display.

        Args:
            indent: Indentation string for each entry.

        Returns:
            Formatted string with all reasoning entries.
        """
        lines = [f"{indent}{entry.key}: {entry.explanation}" for entry in self._entries]
        return "\n".join(lines)


@dataclass(frozen=True)
class LoRAReasoning:
    """Specialized reasoning for LoRA selection decisions.

    Attributes:
        lora_name: Name of the LoRA.
        reason: Explanation for selection.
        confidence: Confidence score (0.0 to 1.0).

    Example:
        >>> reasoning = LoRAReasoning("photorealism_v1", "High priority tag match", 0.85)
        >>> print(reasoning.lora_name, reasoning.confidence)
        photorealism_v1 0.85
    """

    lora_name: str
    reason: str
    confidence: float

    def __post_init__(self) -> None:
        """Validate LoRA reasoning."""
        if not self.lora_name:
            raise ValueError("LoRA name cannot be empty")
        if not self.reason:
            raise ValueError("LoRA reasoning cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

    @property
    def is_high_confidence(self) -> bool:
        """Check if confidence is high (>= 0.7)."""
        return self.confidence >= 0.7

    @property
    def is_low_confidence(self) -> bool:
        """Check if confidence is low (< 0.5)."""
        return self.confidence < 0.5


@dataclass(frozen=True)
class ParameterReasoning:
    """Specialized reasoning for parameter adjustment decisions.

    Attributes:
        parameter: Parameter name (e.g., "steps", "cfg").
        original_value: Original value before adjustment.
        adjusted_value: Value after adjustment.
        reason: Explanation for adjustment.

    Example:
        >>> reasoning = ParameterReasoning("cfg", 7.0, 9.0, "Increased for detail")
        >>> print(reasoning.parameter, reasoning.delta)
        cfg 2.0
    """

    parameter: str
    original_value: float
    adjusted_value: float
    reason: str

    def __post_init__(self) -> None:
        """Validate parameter reasoning."""
        if not self.parameter:
            raise ValueError("Parameter name cannot be empty")
        if not self.reason:
            raise ValueError("Parameter reasoning cannot be empty")

    @property
    def delta(self) -> float:
        """Calculate change in value."""
        return self.adjusted_value - self.original_value

    @property
    def is_increase(self) -> bool:
        """Check if value increased."""
        return self.delta > 0

    @property
    def is_decrease(self) -> bool:
        """Check if value decreased."""
        return self.delta < 0

    @property
    def is_unchanged(self) -> bool:
        """Check if value is unchanged."""
        return abs(self.delta) < 0.001

    @property
    def percent_change(self) -> float:
        """Calculate percent change."""
        if self.original_value == 0:
            return 0.0
        return (self.delta / self.original_value) * 100


__all__ = [
    "ReasoningEntry",
    "ReasoningMap",
    "LoRAReasoning",
    "ParameterReasoning",
]
