"""Weight value objects for neural network weights.

This module provides type-safe weight classes WITHOUT using dicts, tuples, or any.
"""

from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class LoRAWeight:
    """Represents a single LoRA weight with validation.

    Attributes:
        lora_name: Name/identifier of the LoRA.
        weight: Weight value (typically 0.0 to 1.5).

    Example:
        >>> weight = LoRAWeight("photorealism_v1", 0.8)
        >>> print(weight.lora_name, weight.weight)
        photorealism_v1 0.8
    """

    lora_name: str
    weight: float

    def __post_init__(self) -> None:
        """Validate LoRA weight."""
        if not self.lora_name:
            raise ValueError("LoRA name cannot be empty")
        if self.weight < 0:
            raise ValueError(f"Weight cannot be negative, got {self.weight}")
        if self.weight > 2.0:
            raise ValueError(f"Weight too high (max 2.0), got {self.weight}")


@dataclass(frozen=True)
class LoRAWeights:
    """Collection of LoRA weights with validation.

    Attributes:
        _weights: Internal list of LoRAWeight instances (frozen).

    Example:
        >>> weights = LoRAWeights([
        ...     LoRAWeight("photorealism_v1", 0.8),
        ...     LoRAWeight("detail_enhancer", 0.6)
        ... ])
        >>> print(weights.total_weight)
        1.4
        >>> print(weights.count)
        2
    """

    _weights: list[LoRAWeight] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate LoRA weights collection."""
        if not self._weights:
            raise ValueError("LoRAWeights cannot be empty")

        # Check for duplicate LoRA names
        names = [w.lora_name for w in self._weights]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate LoRA names found")

    @property
    def total_weight(self) -> float:
        """Calculate total weight across all LoRAs."""
        return sum(w.weight for w in self._weights)

    @property
    def count(self) -> int:
        """Number of LoRAs."""
        return len(self._weights)

    def get_weight(self, lora_name: str) -> float | None:
        """Get weight for a specific LoRA.

        Args:
            lora_name: Name of the LoRA.

        Returns:
            Weight value or None if not found.
        """
        for w in self._weights:
            if w.lora_name == lora_name:
                return w.weight
        return None

    def has_lora(self, lora_name: str) -> bool:
        """Check if a specific LoRA is present.

        Args:
            lora_name: Name of the LoRA.

        Returns:
            True if present, False otherwise.
        """
        return any(w.lora_name == lora_name for w in self._weights)

    def get_all(self) -> list[LoRAWeight]:
        """Get all LoRA weights.

        Returns:
            List of all LoRAWeight instances.
        """
        return self._weights.copy()

    def __iter__(self) -> Iterator[LoRAWeight]:
        """Iterate over LoRA weights."""
        return iter(self._weights)

    def __len__(self) -> int:
        """Number of LoRAs."""
        return len(self._weights)


@dataclass(frozen=True)
class ParameterDelta:
    """Represents a single parameter delta.

    Attributes:
        parameter: Parameter name.
        delta: Delta value.

    Example:
        >>> delta = ParameterDelta("steps", 5.0)
        >>> print(delta.parameter, delta.delta)
        steps 5.0
    """

    parameter: str
    delta: float

    def __post_init__(self) -> None:
        """Validate parameter delta."""
        if not self.parameter:
            raise ValueError("Parameter name cannot be empty")


@dataclass(frozen=True)
class DeltaWeights:
    """Delta weights for fine-tuning adjustments.

    Attributes:
        _deltas: Internal list of ParameterDelta instances (frozen).

    Example:
        >>> deltas = DeltaWeights([
        ...     ParameterDelta("steps", 5.0),
        ...     ParameterDelta("cfg", 0.5)
        ... ])
        >>> print(deltas.get("steps"))
        5.0
    """

    _deltas: list[ParameterDelta] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate delta weights."""
        if not self._deltas:
            raise ValueError("DeltaWeights cannot be empty")

        # Check for duplicate parameter names
        params = [d.parameter for d in self._deltas]
        if len(params) != len(set(params)):
            raise ValueError("Duplicate parameter names found")

    def get(self, param_name: str, default: float = 0.0) -> float:
        """Get delta for a parameter.

        Args:
            param_name: Name of the parameter.
            default: Default value if not found.

        Returns:
            Delta value.
        """
        for d in self._deltas:
            if d.parameter == param_name:
                return d.delta
        return default

    def has(self, param_name: str) -> bool:
        """Check if a parameter delta exists.

        Args:
            param_name: Name of the parameter.

        Returns:
            True if exists, False otherwise.
        """
        return any(d.parameter == param_name for d in self._deltas)

    @property
    def param_names(self) -> list[str]:
        """List of parameter names."""
        return [d.parameter for d in self._deltas]

    def get_all(self) -> list[ParameterDelta]:
        """Get all parameter deltas.

        Returns:
            List of all ParameterDelta instances.
        """
        return self._deltas.copy()


@dataclass(frozen=True)
class WeightConfig:
    """Configuration for weight limits and validation.

    Attributes:
        min_weight: Minimum allowed weight.
        max_weight: Maximum allowed weight.
        max_total_weight: Maximum total weight across all LoRAs.

    Example:
        >>> config = WeightConfig(min_weight=0.3, max_weight=1.2, max_total_weight=3.0)
        >>> print(config.is_valid_weight(0.5))
        True
    """

    min_weight: float = 0.3
    max_weight: float = 1.2
    max_total_weight: float = 3.0

    def __post_init__(self) -> None:
        """Validate weight config."""
        if self.min_weight < 0:
            raise ValueError(f"min_weight cannot be negative, got {self.min_weight}")
        if self.max_weight <= self.min_weight:
            raise ValueError(
                f"max_weight ({self.max_weight}) must be greater than "
                f"min_weight ({self.min_weight})"
            )
        if self.max_total_weight <= 0:
            raise ValueError(
                f"max_total_weight must be positive, got {self.max_total_weight}"
            )

    def is_valid_weight(self, weight: float) -> bool:
        """Check if a weight is within valid range.

        Args:
            weight: Weight value to validate.

        Returns:
            True if valid, False otherwise.
        """
        return self.min_weight <= weight <= self.max_weight

    def is_valid_total(self, total: float) -> bool:
        """Check if total weight is within limit.

        Args:
            total: Total weight to validate.

        Returns:
            True if valid, False otherwise.
        """
        return total <= self.max_total_weight

    def clamp_weight(self, weight: float) -> float:
        """Clamp weight to valid range.

        Args:
            weight: Weight value to clamp.

        Returns:
            Clamped weight value.
        """
        return max(self.min_weight, min(self.max_weight, weight))


__all__ = ["LoRAWeight", "LoRAWeights", "ParameterDelta", "DeltaWeights", "WeightConfig"]
