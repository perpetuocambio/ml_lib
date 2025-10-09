"""Single raw configuration entry."""

from dataclasses import dataclass

from infrastructure.config.types.config_value import ConfigValue


@dataclass(frozen=True)
class ConfigDataEntry:
    """Single raw configuration entry."""

    key: str
    value: ConfigValue
