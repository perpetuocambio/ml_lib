"""Configuration entity for individual agent roles."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentRoleConfig:
    """Configuration for an individual agent role."""

    display_name: str
    role_id: str
    description: str
    autonomy_level: str
    cost_threshold: int
    available_mcp_tools: list[str]
    specializations: list[str]
    prompt_template_id: str
