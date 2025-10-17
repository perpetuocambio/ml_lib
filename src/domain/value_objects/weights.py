"""Value Objects for weights and scores - Immutable, validated primitives.

These replace raw floats with typed, validated objects that guarantee invariants.
"""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class LoRAWeight:
    """
    LoRA weight/alpha value - typically 0.0 to 2.0.

    Immutable Value Object that guarantees valid range.

    Example:
        >>> weight = LoRAWeight(0.8)
        >>> weight.value
        0.8
        >>> LoRAWeight(-1.0)  # Raises ValueError
    """

    value: float

    # Class constants
    MIN_VALUE: ClassVar[float] = 0.0
    MAX_VALUE: ClassVar[float] = 2.0
    DEFAULT: ClassVar[float] = 1.0

    def __post_init__(self):
        """Validate on construction."""
        if not (self.MIN_VALUE <= self.value <= self.MAX_VALUE):
            raise ValueError(
                f"LoRAWeight must be between {self.MIN_VALUE} and {self.MAX_VALUE}, "
                f"got {self.value}"
            )

    @classmethod
    def default(cls) -> "LoRAWeight":
        """Create default weight (1.0)."""
        return cls(cls.DEFAULT)

    @classmethod
    def from_float(cls, value: float) -> "LoRAWeight":
        """Create from float, clamping to valid range."""
        clamped = max(cls.MIN_VALUE, min(cls.MAX_VALUE, value))
        return cls(clamped)

    def scale_by(self, factor: float) -> "LoRAWeight":
        """
        Scale weight by factor, clamping result.

        Args:
            factor: Multiplier

        Returns:
            New LoRAWeight with scaled value
        """
        return LoRAWeight.from_float(self.value * factor)

    def __float__(self) -> float:
        """Convert to float."""
        return self.value

    def __str__(self) -> str:
        """String representation."""
        return f"{self.value:.2f}"


@dataclass(frozen=True)
class PromptWeight:
    """
    Prompt emphasis weight - typically 0.5 to 1.5.

    Used for (word:weight) syntax in prompts.
    """

    value: float

    MIN_VALUE: ClassVar[float] = 0.1
    MAX_VALUE: ClassVar[float] = 2.0
    DEFAULT: ClassVar[float] = 1.0

    def __post_init__(self):
        """Validate on construction."""
        if not (self.MIN_VALUE <= self.value <= self.MAX_VALUE):
            raise ValueError(
                f"PromptWeight must be between {self.MIN_VALUE} and {self.MAX_VALUE}, "
                f"got {self.value}"
            )

    @classmethod
    def default(cls) -> "PromptWeight":
        """Create default weight (1.0)."""
        return cls(cls.DEFAULT)

    @classmethod
    def emphasized(cls) -> "PromptWeight":
        """Create emphasized weight (1.1)."""
        return cls(1.1)

    @classmethod
    def strongly_emphasized(cls) -> "PromptWeight":
        """Create strongly emphasized weight (1.21)."""
        return cls(1.21)

    @classmethod
    def de_emphasized(cls) -> "PromptWeight":
        """Create de-emphasized weight (0.9)."""
        return cls(0.9)

    def is_emphasized(self) -> bool:
        """Check if this is an emphasis (> 1.0)."""
        return self.value > 1.0

    def is_default(self) -> bool:
        """Check if this is default weight."""
        return abs(self.value - self.DEFAULT) < 0.01

    def __float__(self) -> float:
        """Convert to float."""
        return self.value

    def __str__(self) -> str:
        """String representation."""
        return f"{self.value:.2f}"


@dataclass(frozen=True)
class ConfidenceScore:
    """
    Confidence score - 0.0 to 1.0.

    Represents confidence in recommendations, predictions, etc.
    """

    value: float

    MIN_VALUE: ClassVar[float] = 0.0
    MAX_VALUE: ClassVar[float] = 1.0

    def __post_init__(self):
        """Validate on construction."""
        if not (self.MIN_VALUE <= self.value <= self.MAX_VALUE):
            raise ValueError(
                f"ConfidenceScore must be between {self.MIN_VALUE} and {self.MAX_VALUE}, "
                f"got {self.value}"
            )

    @classmethod
    def from_percentage(cls, percentage: float) -> "ConfidenceScore":
        """
        Create from percentage (0-100).

        Args:
            percentage: Value from 0 to 100

        Returns:
            ConfidenceScore normalized to 0-1
        """
        if not (0 <= percentage <= 100):
            raise ValueError(f"Percentage must be 0-100, got {percentage}")
        return cls(percentage / 100.0)

    @classmethod
    def low(cls) -> "ConfidenceScore":
        """Create low confidence (0.3)."""
        return cls(0.3)

    @classmethod
    def medium(cls) -> "ConfidenceScore":
        """Create medium confidence (0.5)."""
        return cls(0.5)

    @classmethod
    def high(cls) -> "ConfidenceScore":
        """Create high confidence (0.8)."""
        return cls(0.8)

    @classmethod
    def very_high(cls) -> "ConfidenceScore":
        """Create very high confidence (0.95)."""
        return cls(0.95)

    def is_low(self) -> bool:
        """Check if confidence is low (< 0.4)."""
        return self.value < 0.4

    def is_medium(self) -> bool:
        """Check if confidence is medium (0.4-0.7)."""
        return 0.4 <= self.value < 0.7

    def is_high(self) -> bool:
        """Check if confidence is high (>= 0.7)."""
        return self.value >= 0.7

    def to_percentage(self) -> float:
        """Convert to percentage (0-100)."""
        return self.value * 100.0

    def __float__(self) -> float:
        """Convert to float."""
        return self.value

    def __str__(self) -> str:
        """String representation."""
        return f"{self.to_percentage():.1f}%"


@dataclass(frozen=True)
class CFGScale:
    """
    Classifier-Free Guidance scale - typically 1.0 to 20.0.

    Controls how strongly the model follows the prompt.
    """

    value: float

    MIN_VALUE: ClassVar[float] = 1.0
    MAX_VALUE: ClassVar[float] = 30.0
    DEFAULT_SDXL: ClassVar[float] = 7.0
    DEFAULT_PONY: ClassVar[float] = 6.0
    DEFAULT_SD15: ClassVar[float] = 7.5

    def __post_init__(self):
        """Validate on construction."""
        if not (self.MIN_VALUE <= self.value <= self.MAX_VALUE):
            raise ValueError(
                f"CFGScale must be between {self.MIN_VALUE} and {self.MAX_VALUE}, "
                f"got {self.value}"
            )

    @classmethod
    def default_for_model(cls, model_type: str) -> "CFGScale":
        """
        Get default CFG for model type.

        Args:
            model_type: "sdxl", "pony", "sd15", etc.

        Returns:
            Appropriate default CFG
        """
        model_lower = model_type.lower()
        if "pony" in model_lower:
            return cls(cls.DEFAULT_PONY)
        elif "sdxl" in model_lower or "xl" in model_lower:
            return cls(cls.DEFAULT_SDXL)
        elif "1.5" in model_lower or "sd15" in model_lower:
            return cls(cls.DEFAULT_SD15)
        else:
            return cls(cls.DEFAULT_SDXL)  # Fallback

    def is_low(self) -> bool:
        """Check if CFG is low (< 5.0) - more creative."""
        return self.value < 5.0

    def is_high(self) -> bool:
        """Check if CFG is high (> 10.0) - more adherent."""
        return self.value > 10.0

    def __float__(self) -> float:
        """Convert to float."""
        return self.value

    def __str__(self) -> str:
        """String representation."""
        return f"{self.value:.1f}"
