"""
Data structure for prompt version persistence.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptVersionData:
    """Data structure for prompt version database records."""

    version_id: str
    agent_id: str
    prompt_content: str
    version_number: int
    change_description: str | None
    created_at: str
    is_active: bool
