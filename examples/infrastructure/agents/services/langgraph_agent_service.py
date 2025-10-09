"""LangGraph agent service implementation."""

import asyncio
from collections.abc import AsyncGenerator
from uuid import UUID

from infrastructure.agents.state_management.agent_state_manager import AgentStateManager
from infrastructure.agents.types.active_agent_cache import ActiveAgentCache
from langgraph.graph import StateGraph


class LangGraphAgentService:
    """Service for managing LangGraph agents."""

    def __init__(self, state_manager: AgentStateManager):
        """Initialize with state manager."""
        self.state_manager = state_manager
        self.active_agents = ActiveAgentCache()

    async def create_agent(
        self, project_id: UUID, agent_type: str, system_prompt: str
    ) -> UUID:
        """Create and initialize a new LangGraph agent."""
        # Create basic LangGraph workflow
        workflow = StateGraph({})

        # Initialize agent state in database
        agent_id = await self.state_manager.create_agent_state(
            project_id=project_id,
            agent_type=agent_type,
            initial_state=f'{{"system_prompt": "{system_prompt}", "status": "CREATED"}}',
        )

        # Store active agent
        self.active_agents.put(agent_id, workflow.compile())

        return agent_id

    async def send_message_to_agent(
        self, agent_id: UUID, message: str
    ) -> AsyncGenerator[str, None]:
        """Send message to agent and stream response."""
        if not self.active_agents.has_agent(agent_id):
            raise ValueError(f"Agent {agent_id} not found")

        # Simple implementation - in real scenario this would be more complex
        # agent = self.active_agents.get(agent_id) # Would be used for actual LangGraph interaction
        # This is a placeholder for actual LangGraph agent interaction
        response_chunks = [
            f"Agent {agent_id} received: {message}",
            f"Agent {agent_id} is processing...",
            f"Agent {agent_id} completed processing",
        ]

        for chunk in response_chunks:
            yield chunk
            await asyncio.sleep(0.1)  # Simulate streaming

    async def terminate_agent(self, agent_id: UUID) -> bool:
        """Terminate an active agent."""
        if self.active_agents.has_agent(agent_id):
            self.active_agents.remove(agent_id)
            return await self.state_manager.delete_agent_state(agent_id)
        return False

    def get_active_agent_count(self) -> int:
        """Get count of currently active agents."""
        return self.active_agents.get_agent_count()
