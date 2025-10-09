"""Registry data structure for config types."""

from dataclasses import dataclass

from infrastructure.config.base.base_config import BaseInfrastructureConfig


@dataclass
class ConfigRegistryData:
    """Registry data structure for config types."""

    config_type: str
    config_class: type[BaseInfrastructureConfig]
