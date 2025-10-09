"""Type-safe validation value wrapper."""

from dataclasses import dataclass
from enum import Enum

from infrastructure.config.types.enum_value import EnumValue
from infrastructure.config.types.validation_value_type import ValidationValueType


@dataclass(frozen=True)
class ValidationValue:
    """Type-safe wrapper for validation values - eliminates Any."""

    value_type: ValidationValueType
    string_value: str | None = None
    integer_value: int | None = None
    float_value: float | None = None
    boolean_value: bool | None = None
    enum_value: EnumValue | None = None

    @classmethod
    def from_string(cls, value: str) -> "ValidationValue":
        """Create from string value."""
        return cls(value_type=ValidationValueType.STRING, string_value=value)

    @classmethod
    def from_integer(cls, value: int) -> "ValidationValue":
        """Create from integer value."""
        return cls(value_type=ValidationValueType.INTEGER, integer_value=value)

    @classmethod
    def from_float(cls, value: float) -> "ValidationValue":
        """Create from float value."""
        return cls(value_type=ValidationValueType.FLOAT, float_value=value)

    @classmethod
    def from_boolean(cls, value: bool) -> "ValidationValue":
        """Create from boolean value."""
        return cls(value_type=ValidationValueType.BOOLEAN, boolean_value=value)

    @classmethod
    def from_enum(cls, value: Enum) -> "ValidationValue":
        """Create from enum value."""
        enum_wrapper = EnumValue.from_enum(value)
        return cls(value_type=ValidationValueType.ENUM, enum_value=enum_wrapper)

    @classmethod
    def null(cls) -> "ValidationValue":
        """Create null value."""
        return cls(value_type=ValidationValueType.NULL)

    @classmethod
    def from_unknown(
        cls, value: str | int | float | bool | Enum | None
    ) -> "ValidationValue":
        """Create from unknown value type."""
        if value is None:
            return cls.null()
        elif isinstance(value, str):
            return cls.from_string(value)
        elif isinstance(value, int):
            return cls.from_integer(value)
        elif isinstance(value, float):
            return cls.from_float(value)
        elif isinstance(value, bool):
            return cls.from_boolean(value)
        elif isinstance(value, Enum):
            return cls.from_enum(value)
        else:
            # Convert unknown types to string representation
            return cls.from_string(str(value))

    def get_raw_value(self) -> str | int | float | bool | EnumValue | None:
        """Get the raw value for processing."""
        match self.value_type:
            case ValidationValueType.STRING:
                return self.string_value
            case ValidationValueType.INTEGER:
                return self.integer_value
            case ValidationValueType.FLOAT:
                return self.float_value
            case ValidationValueType.BOOLEAN:
                return self.boolean_value
            case ValidationValueType.ENUM:
                return self.enum_value
            case ValidationValueType.NULL:
                return None
