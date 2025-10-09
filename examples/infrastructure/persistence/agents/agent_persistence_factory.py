"""Factory for agent persistence implementations."""

from infrastructure.persistence.agents.infrastructure_agent_repository import (
    SQLiteAgentPersistence,
)
from infrastructure.persistence.agents.interfaces.agent_persistence_interface import (
    IAgentPersistence,
)
from infrastructure.persistence.agents.postgresql_agent_persistence import (
    PostgreSQLAgentPersistence,
)


class AgentPersistenceFactory:
    """Factory for creating agent persistence implementations."""

    @staticmethod
    def create_sqlite(db_path: str) -> IAgentPersistence:
        """Create SQLite agent persistence."""
        return SQLiteAgentPersistence(db_path)

    @staticmethod
    def create_postgresql(connection_string: str) -> IAgentPersistence:
        """Create PostgreSQL agent persistence."""
        return PostgreSQLAgentPersistence(connection_string)

    @staticmethod
    def create_from_config(persistence_type: str, **kwargs) -> IAgentPersistence:
        """Create persistence based on configuration."""
        if persistence_type.lower() == "sqlite":
            return AgentPersistenceFactory.create_sqlite(kwargs["db_path"])
        elif persistence_type.lower() == "postgresql":
            return AgentPersistenceFactory.create_postgresql(
                kwargs["connection_string"]
            )
        else:
            raise ValueError(f"Unsupported persistence type: {persistence_type}")
