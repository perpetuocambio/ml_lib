"""Type-safe configuration value."""

from dataclasses import dataclass

from infrastructure.config.types.config_data_entry import ConfigDataEntry
from infrastructure.config.types.config_value_type import ConfigValueType
from infrastructure.serialization.protocol_serializer import ProtocolSerializer


@dataclass(frozen=True)
class ConfigValue:
    """Type-safe configuration value - eliminates Any."""

    value_type: ConfigValueType
    string_value: str | None = None
    integer_value: int | None = None
    float_value: float | None = None
    boolean_value: bool | None = None
    list_value: list[str] | None = None
    nested_entries: list[ConfigDataEntry] | None = None

    @classmethod
    def from_string(cls, value: str) -> "ConfigValue":
        """Create from string."""
        return cls(value_type=ConfigValueType.STRING, string_value=value)

    @classmethod
    def from_integer(cls, value: int) -> "ConfigValue":
        """Create from integer."""
        return cls(value_type=ConfigValueType.INTEGER, integer_value=value)

    @classmethod
    def from_float(cls, value: float) -> "ConfigValue":
        """Create from float."""
        return cls(value_type=ConfigValueType.FLOAT, float_value=value)

    @classmethod
    def from_boolean(cls, value: bool) -> "ConfigValue":
        """Create from boolean."""
        return cls(value_type=ConfigValueType.BOOLEAN, boolean_value=value)

    @classmethod
    def from_list(cls, value: list[str]) -> "ConfigValue":
        """Create from string list."""
        return cls(value_type=ConfigValueType.LIST, list_value=value)

    @classmethod
    def null(cls) -> "ConfigValue":
        """Create null value."""
        return cls(value_type=ConfigValueType.NULL)

    def to_protocol_value(
        self,
    ) -> (
        str
        | int
        | float
        | bool
        | list[str]
        | ProtocolSerializer.ConfigProtocolValue
        | None
    ):
        """Convert to protocol value for file loading boundary."""
        match self.value_type:
            case ConfigValueType.STRING:
                return self.string_value
            case ConfigValueType.INTEGER:
                return self.integer_value
            case ConfigValueType.FLOAT:
                return self.float_value
            case ConfigValueType.BOOLEAN:
                return self.boolean_value
            case ConfigValueType.LIST:
                return self.list_value
            case ConfigValueType.NESTED:
                if self.nested_entries:
                    return ProtocolSerializer.serialize_nested_config(
                        {
                            entry.key: entry.value.to_protocol_value()
                            for entry in self.nested_entries
                        }
                    )
                return ProtocolSerializer.serialize_empty_config()
            case ConfigValueType.NULL:
                return None
