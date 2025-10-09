"""Single active agents entry."""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class ActiveAgentsEntry:
    """Single active agents entry."""

    project_id: UUID
    agent_ids: list[UUID]
