"""Entity for proposed agent actions to replace Dict usage."""

from dataclasses import dataclass


@dataclass
class ProposedAction:
    """
    Agent proposed action data.
    Replaces generic dict violations.
    """

    action_type: str
    reasoning: str
    justification: str
    estimated_cost: int | None = None
    can_auto_approve: bool | None = None
    risk_level: str | None = None
    is_routine: bool | None = None
