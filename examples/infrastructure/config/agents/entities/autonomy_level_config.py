"""Configuration entity for autonomy levels."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AutonomyLevelConfig:
    """Configuration for an autonomy level."""

    display_name: str
    can_auto_approve: bool
    max_cost_threshold: int
    requires_supervision: bool
