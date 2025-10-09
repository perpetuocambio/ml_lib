"""Agent data transfer object for Infrastructure layer."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class AgentData:
    """Data transfer object for agent persistence."""

    agent_id: str
    name: str
    role_name: str
    role_description: str
    system_prompt: str
    preferred_tools: list[str]
    expertise_domains: list[str]
    reasoning_style: str
    autonomy_level: str
    collaboration_style: str
    confidence_threshold: float
    max_concurrent_tasks: int
    status: str
    created_at: datetime
    last_active_at: datetime
    project_id: str
    user_id: str
