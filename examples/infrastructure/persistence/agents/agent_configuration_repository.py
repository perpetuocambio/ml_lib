"""
Simple SQLite repository for agent configurations.

Dynamic Agent Configuration System
Direct SQLite implementation avoiding complex config dependencies.
"""

import json
import sqlite3

from infrastructure.persistence.agents.agent_configuration_data import (
    AgentConfigurationData,
)


class AgentConfigurationRepository:
    """Simple repository for agent configurations using direct SQLite."""

    def __init__(self, db_path: str = "pyintelcivil.db"):
        """Initialize repository with database path."""
        self.db_path = db_path

    def create_agent_configuration(self, config_data: AgentConfigurationData) -> bool:
        """Create new agent configuration."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO agent_configurations
                    (agent_id, user_id, name, role_description, system_prompt,
                     capabilities, autonomy_level, knowledge_context, is_active,
                     created_at, last_modified, performance_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        config_data.agent_id,
                        config_data.user_id,
                        config_data.name,
                        config_data.role_description,
                        config_data.system_prompt,
                        json.dumps(config_data.capabilities),
                        config_data.autonomy_level,
                        config_data.knowledge_context,
                        1 if config_data.is_active else 0,
                        config_data.created_at,
                        config_data.last_modified,
                        config_data.performance_metrics_json,
                    ),
                )
                return True
        except Exception as e:
            print(f"Error creating agent configuration: {e}")
            return False

    def get_agent_configuration(self, agent_id: str) -> AgentConfigurationData | None:
        """Retrieve agent configuration by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT agent_id, user_id, name, role_description, system_prompt,
                           capabilities, autonomy_level, knowledge_context, is_active,
                           created_at, last_modified, performance_metrics
                    FROM agent_configurations
                    WHERE agent_id = ?
                """,
                    (agent_id,),
                )

                row = cursor.fetchone()
                if row:
                    return AgentConfigurationData(
                        agent_id=row[0],
                        user_id=row[1],
                        name=row[2],
                        role_description=row[3],
                        system_prompt=row[4],
                        capabilities=json.loads(row[5]),
                        autonomy_level=row[6],
                        knowledge_context=row[7],
                        is_active=bool(row[8]),
                        created_at=row[9],
                        last_modified=row[10],
                        performance_metrics_json=row[11],
                    )
                return None
        except Exception as e:
            print(f"Error retrieving agent configuration: {e}")
            return None

    def list_user_agent_configurations(
        self, user_id: str
    ) -> list[AgentConfigurationData]:
        """List all agent configurations for a user."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT agent_id, user_id, name, role_description, system_prompt,
                           capabilities, autonomy_level, knowledge_context, is_active,
                           created_at, last_modified, performance_metrics
                    FROM agent_configurations
                    WHERE user_id = ?
                    ORDER BY last_modified DESC
                """,
                    (user_id,),
                )

                results = []
                for row in cursor.fetchall():
                    results.append(
                        AgentConfigurationData(
                            agent_id=row[0],
                            user_id=row[1],
                            name=row[2],
                            role_description=row[3],
                            system_prompt=row[4],
                            capabilities=json.loads(row[5]),
                            autonomy_level=row[6],
                            knowledge_context=row[7],
                            is_active=bool(row[8]),
                            created_at=row[9],
                            last_modified=row[10],
                            performance_metrics_json=row[11],
                        )
                    )
                return results
        except Exception as e:
            print(f"Error listing agent configurations: {e}")
            return []

    def update_agent_configuration(self, config_data: AgentConfigurationData) -> bool:
        """Update existing agent configuration."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE agent_configurations
                    SET name = ?, role_description = ?, system_prompt = ?,
                        capabilities = ?, autonomy_level = ?, knowledge_context = ?,
                        is_active = ?, last_modified = ?, performance_metrics = ?
                    WHERE agent_id = ?
                """,
                    (
                        config_data.name,
                        config_data.role_description,
                        config_data.system_prompt,
                        json.dumps(config_data.capabilities),
                        config_data.autonomy_level,
                        config_data.knowledge_context,
                        1 if config_data.is_active else 0,
                        config_data.last_modified,
                        config_data.performance_metrics_json,
                        config_data.agent_id,
                    ),
                )
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error updating agent configuration: {e}")
            return False

    def delete_agent_configuration(self, agent_id: str) -> bool:
        """Delete agent configuration by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM agent_configurations WHERE agent_id = ?", (agent_id,)
                )
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting agent configuration: {e}")
            return False
