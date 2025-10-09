"""Agent role mapping entity."""

from dataclasses import dataclass

from infrastructure.config.agents.entities.agent_role_config import AgentRoleConfig


@dataclass(frozen=True)
class AgentRoleMapping:
    """Mapping of role name to configuration."""

    role_name: str
    config: AgentRoleConfig
