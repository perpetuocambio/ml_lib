"""Domain interface for LLM client."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ILLMClient(ABC):
    """Interface for LLM client interactions."""

    @abstractmethod
    async def generate_agent_reasoning(self, agent_id: str, context_data: str) -> str:
        """Generate reasoning for an agent based on context.

        Args:
            agent_id: The ID of the agent
            context_data: Current context information

        Returns:
            str: Generated reasoning text
        """

    @abstractmethod
    async def evaluate_proposal_priority(self, proposal_data: str) -> str:
        """Evaluate the priority of a proposal.

        Args:
            proposal_data: Data about the proposal to evaluate

        Returns:
            str: Priority evaluation result
        """
