"""Autonomy level mapping entity."""

from dataclasses import dataclass

from infrastructure.config.agents.entities.autonomy_level_config import (
    AutonomyLevelConfig,
)


@dataclass(frozen=True)
class AutonomyLevelMapping:
    """Mapping of autonomy level name to configuration."""

    level_name: str
    config: AutonomyLevelConfig
