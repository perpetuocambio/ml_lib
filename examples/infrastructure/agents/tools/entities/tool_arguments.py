"""Tool arguments value object for domain operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from infrastructure.serialization.protocol_serializer import ProtocolSerializer

# Type alias for tool arguments (eliminates generic dict)
ToolArgumentsDict = ProtocolSerializer.ToolArgumentsType

T = TypeVar("T")


@dataclass(frozen=True)
class ToolArguments:
    """Type-safe tool arguments."""

    arguments: ToolArgumentsDict

    def __post_init__(self) -> None:
        """Validate tool arguments."""
        if not ProtocolSerializer.is_valid_tool_arguments(self.arguments):
            raise ValueError("Arguments must be a dictionary")
        if not all(
            isinstance(k, str) and isinstance(v, str) for k, v in self.arguments.items()
        ):
            raise ValueError("All keys and values must be strings")

    @classmethod
    def create(cls, arguments: ToolArgumentsDict) -> ToolArguments:
        """Create tool arguments from typed dict."""
        return cls(arguments=arguments)

    @classmethod
    def empty(cls) -> ToolArguments:
        """Create empty tool arguments."""
        return cls(arguments=ProtocolSerializer.serialize_empty_arguments())

    def get(self, key: str) -> str | None:
        """Get argument by key."""
        return self.arguments.get(key)

    def has(self, key: str) -> bool:
        """Check if argument exists."""
        return key in self.arguments

    def convert_to(self, target_type: type[T]) -> T:
        """Convert tool arguments to specific typed object."""
        # If target type is a dataclass, use field mapping
        if hasattr(target_type, "__dataclass_fields__"):
            return self._convert_to_dataclass(target_type)

        # If target type has from_tool_arguments method
        if hasattr(target_type, "from_tool_arguments"):
            return target_type.from_tool_arguments(self.arguments)

        raise ValueError(f"Cannot convert ToolArguments to {target_type}")

    def _convert_to_dataclass(self, target_type: type[T]) -> T:
        """Convert to dataclass using field mapping."""
        kwargs = {}
        for field_name, field_info in target_type.__dataclass_fields__.items():
            if field_name in self.arguments:
                # Convert string to appropriate type
                value = self.arguments[field_name]
                field_type = field_info.type

                # Basic type conversion
                if field_type is int:
                    kwargs[field_name] = int(value)
                elif field_type is float:
                    kwargs[field_name] = float(value)
                elif field_type is bool:
                    kwargs[field_name] = value.lower() in ("true", "1", "yes")
                else:
                    kwargs[field_name] = value

        return target_type(**kwargs)

    def get_all_arguments(self) -> list[tuple[str, str]]:
        """Get all arguments as type-safe tuples."""
        return ProtocolSerializer.get_tool_argument_items(self.arguments)

    def keys(self) -> list[str]:
        """Get all argument keys."""
        return ProtocolSerializer.get_tool_argument_keys(self.arguments)

    def values(self) -> list[str]:
        """Get all argument values."""
        return ProtocolSerializer.get_tool_argument_values(self.arguments)

    def is_empty(self) -> bool:
        """Check if arguments are empty."""
        return len(self.arguments) == 0
