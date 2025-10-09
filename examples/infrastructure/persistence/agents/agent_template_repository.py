"""
Simple SQLite repository for agent templates.

Dynamic Agent Configuration System
Direct SQLite implementation for template management.
"""

import json
import sqlite3

from infrastructure.persistence.agents.agent_template_data import AgentTemplateData


class AgentTemplateRepository:
    """Simple repository for agent templates using direct SQLite."""

    def __init__(self, db_path: str = "pyintelcivil.db"):
        """Initialize repository with database path."""
        self.db_path = db_path

    def get_template(self, template_id: str) -> AgentTemplateData | None:
        """Retrieve agent template by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT template_id, name, description, role_archetype, base_system_prompt,
                           default_capabilities, suggested_autonomy, usage_examples, category,
                           complexity_level, is_official, created_at, usage_count, avg_rating
                    FROM agent_templates
                    WHERE template_id = ?
                """,
                    (template_id,),
                )

                row = cursor.fetchone()
                if row:
                    return AgentTemplateData(
                        template_id=row[0],
                        name=row[1],
                        description=row[2],
                        role_archetype=row[3],
                        base_system_prompt=row[4],
                        default_capabilities=json.loads(row[5]),
                        suggested_autonomy=row[6],
                        usage_examples=row[7],
                        category=row[8],
                        complexity_level=row[9],
                        is_official=bool(row[10]),
                        created_at=row[11],
                        usage_count=row[12],
                        avg_rating=row[13],
                    )
                return None
        except Exception as e:
            print(f"Error retrieving agent template: {e}")
            return None

    def list_templates_by_category(
        self, category: str = None
    ) -> list[AgentTemplateData]:
        """List agent templates, optionally filtered by category."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if category:
                    cursor.execute(
                        """
                        SELECT template_id, name, description, role_archetype, base_system_prompt,
                               default_capabilities, suggested_autonomy, usage_examples, category,
                               complexity_level, is_official, created_at, usage_count, avg_rating
                        FROM agent_templates
                        WHERE category = ?
                        ORDER BY avg_rating DESC, usage_count DESC
                    """,
                        (category,),
                    )
                else:
                    cursor.execute("""
                        SELECT template_id, name, description, role_archetype, base_system_prompt,
                               default_capabilities, suggested_autonomy, usage_examples, category,
                               complexity_level, is_official, created_at, usage_count, avg_rating
                        FROM agent_templates
                        ORDER BY avg_rating DESC, usage_count DESC
                    """)

                results = []
                for row in cursor.fetchall():
                    results.append(
                        AgentTemplateData(
                            template_id=row[0],
                            name=row[1],
                            description=row[2],
                            role_archetype=row[3],
                            base_system_prompt=row[4],
                            default_capabilities=json.loads(row[5]),
                            suggested_autonomy=row[6],
                            usage_examples=row[7],
                            category=row[8],
                            complexity_level=row[9],
                            is_official=bool(row[10]),
                            created_at=row[11],
                            usage_count=row[12],
                            avg_rating=row[13],
                        )
                    )
                return results
        except Exception as e:
            print(f"Error listing agent templates: {e}")
            return []

    def increment_template_usage(self, template_id: str) -> bool:
        """Increment usage count for a template."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE agent_templates
                    SET usage_count = usage_count + 1
                    WHERE template_id = ?
                """,
                    (template_id,),
                )
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error incrementing template usage: {e}")
            return False

    def get_available_capabilities(self) -> list[str]:
        """Get list of available agent capabilities."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM agent_capabilities ORDER BY name")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error retrieving capabilities: {e}")
            return []

    def get_autonomy_levels(self) -> list[str]:
        """Get list of available autonomy levels."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM agent_autonomy_levels ORDER BY approval_threshold DESC"
                )
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error retrieving autonomy levels: {e}")
            return []
