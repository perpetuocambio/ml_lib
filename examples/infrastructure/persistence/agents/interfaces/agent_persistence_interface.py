"""Infrastructure persistence interface for agents."""

from abc import ABC, abstractmethod

from infrastructure.persistence.agents.agent_data import AgentData


class IAgentPersistence(ABC):
    """Infrastructure interface for agent persistence - allows multiple implementations."""

    @abstractmethod
    def save_agent(self, agent_data: AgentData) -> None:
        """Save or update an agent."""
        pass

    @abstractmethod
    def get_agent_by_id(self, agent_id: str) -> AgentData | None:
        """Get agent by ID."""
        pass

    @abstractmethod
    def get_agents_by_project_id(self, project_id: str) -> list[AgentData]:
        """Get all agents for a specific project."""
        pass

    @abstractmethod
    def get_agents_by_user_id(self, user_id: str) -> list[AgentData]:
        """Get all agents for a specific user."""
        pass

    @abstractmethod
    def delete_agent(self, agent_id: str) -> bool:
        """Delete agent by ID."""
        pass

    @abstractmethod
    def agent_exists(self, agent_id: str) -> bool:
        """Check if agent exists."""
        pass
