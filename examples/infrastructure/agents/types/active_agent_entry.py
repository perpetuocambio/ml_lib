"""Single active agent entry."""

from dataclasses import dataclass
from uuid import UUID

from langgraph.graph import StateGraph


@dataclass(frozen=True)
class ActiveAgentEntry:
    """Single active agent entry."""

    agent_id: UUID
    state_graph: StateGraph
