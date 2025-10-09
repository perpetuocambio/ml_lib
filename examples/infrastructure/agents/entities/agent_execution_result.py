"""Entity for agent execution results to replace Dict usage."""

from dataclasses import dataclass

from infrastructure.agents.entities.proposed_action import ProposedAction


@dataclass(frozen=True)
class AgentExecutionResult:
    """
    Agent execution result data.
    Replaces generic dict violations.
    """

    success: bool
    agent_id: str
    phase: str
    observations_count: int | None = None
    action_taken: bool | None = None
    proposed_action: ProposedAction | None = None
    error: str | None = None
