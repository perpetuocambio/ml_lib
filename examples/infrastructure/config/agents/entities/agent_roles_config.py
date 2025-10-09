"""Configuration entity for all agent roles and related configs."""

from dataclasses import dataclass

from infrastructure.config.agents.entities.agent_role_mapping import AgentRoleMapping
from infrastructure.config.agents.entities.autonomy_level_mapping import (
    AutonomyLevelMapping,
)
from infrastructure.config.agents.entities.tool_cost_mapping import ToolCostMapping


@dataclass(frozen=True)
class AgentRolesConfig:
    """Complete configuration for agent roles system."""

    roles: list[AgentRoleMapping]
    autonomy_levels: list[AutonomyLevelMapping]
    tool_costs: list[ToolCostMapping]
