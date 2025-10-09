"""
Data structure for agent template persistence.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentTemplateData:
    """Data structure for agent template persistence."""

    template_id: str
    name: str
    description: str
    role_archetype: str
    base_system_prompt: str
    default_capabilities: list[str]
    suggested_autonomy: str
    usage_examples: str
    category: str
    complexity_level: str
    is_official: bool
    created_at: str
    usage_count: int
    avg_rating: float
