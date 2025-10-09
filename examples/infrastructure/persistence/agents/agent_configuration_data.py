"""
Data structure for agent configuration persistence.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfigurationData:
    """Data structure for agent configuration persistence."""

    agent_id: str
    user_id: str
    name: str
    role_description: str
    system_prompt: str
    capabilities: list[str]
    autonomy_level: str
    knowledge_context: str
    is_active: bool
    created_at: str
    last_modified: str
    performance_metrics_json: str
