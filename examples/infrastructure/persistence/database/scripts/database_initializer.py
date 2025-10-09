"""Database initialization for PostgreSQL with pgvector setup."""

import asyncio
import logging
from datetime import datetime

from infrastructure.persistence.database.connections.postgresql_connection import (
    PostgreSQLConnection,
)
from infrastructure.persistence.database.database_health_status import (
    DatabaseHealthStatus,
)

logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """
    Database initialization service for PostgreSQL with pgvector.

    Handles schema creation, extensions setup, and initial data loading.
    """

    def __init__(self, db_connection: PostgreSQLConnection):
        """Initialize database initializer."""
        self.db_connection = db_connection

    async def initialize_database(self) -> bool:
        """
        Initialize complete database schema and extensions.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Starting database initialization...")

            # Initialize connection pool
            await self.db_connection.initialize_pool()

            # Create extensions
            await self._create_extensions()

            # Setup vector tables for RAG
            await self.db_connection.setup_vector_tables()

            # Create additional application tables
            await self._create_application_tables()

            # Create indexes for performance
            await self._create_indexes()

            logger.info("Database initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            return False

    async def _create_extensions(self) -> None:
        """Create required PostgreSQL extensions."""
        async with self.db_connection.get_connection() as conn:
            # Enable pgvector for embeddings
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Enable PostGIS for geographic data (future)
            await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")

            # Enable UUID generation
            await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

            logger.info("Database extensions created")

    async def _create_application_tables(self) -> None:
        """Create application-specific tables."""
        async with self.db_connection.get_connection() as conn:
            # Projects table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'::jsonb
                );
            """)

            # Search history table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS search_history (
                    id SERIAL PRIMARY KEY,
                    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
                    query_text TEXT NOT NULL,
                    search_type VARCHAR(50) NOT NULL,
                    results_count INTEGER DEFAULT 0,
                    execution_time_ms INTEGER,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'::jsonb
                );
            """)

            # Document store table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
                    filename VARCHAR(255) NOT NULL,
                    file_path TEXT NOT NULL,
                    mime_type VARCHAR(100),
                    file_size_bytes BIGINT,
                    content_hash VARCHAR(64),
                    processed_at TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'::jsonb
                );
            """)

            # Analysis results table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
                    technique_name VARCHAR(100) NOT NULL,
                    input_data JSONB NOT NULL,
                    output_data JSONB NOT NULL,
                    execution_time_ms INTEGER,
                    confidence_score DECIMAL(3,2),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'::jsonb
                );
            """)

            # Agent states table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_states (
                    agent_id VARCHAR(255) PRIMARY KEY,
                    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
                    role_id VARCHAR(100) NOT NULL,
                    current_phase VARCHAR(50) NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT true,
                    auto_approve_threshold INTEGER NOT NULL DEFAULT 5,
                    performance_score DECIMAL(3,2) NOT NULL DEFAULT 0.0,
                    actions_proposed INTEGER NOT NULL DEFAULT 0,
                    actions_approved INTEGER NOT NULL DEFAULT 0,
                    actions_executed INTEGER NOT NULL DEFAULT 0,
                    last_action_timestamp TIMESTAMP WITH TIME ZONE,
                    recent_observations JSONB DEFAULT '[]'::jsonb,
                    pending_proposals JSONB DEFAULT '[]'::jsonb,
                    completed_actions JSONB DEFAULT '[]'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)

            # Agent proposals table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_proposals (
                    proposal_id VARCHAR(255) PRIMARY KEY,
                    agent_id VARCHAR(255) NOT NULL REFERENCES agent_states(agent_id) ON DELETE CASCADE,
                    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
                    agent_role VARCHAR(100) NOT NULL,
                    action_type VARCHAR(50) NOT NULL,
                    action_description TEXT NOT NULL,
                    mcp_tool_name VARCHAR(100) NOT NULL,
                    reasoning TEXT NOT NULL,
                    expected_outcome TEXT NOT NULL,
                    methodological_justification TEXT NOT NULL,
                    priority VARCHAR(20) NOT NULL,
                    estimated_cost INTEGER NOT NULL,
                    estimated_duration_minutes INTEGER NOT NULL,
                    risk_level VARCHAR(20) NOT NULL,
                    is_routine BOOLEAN NOT NULL DEFAULT false,
                    current_project_state_summary TEXT NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    tool_arguments JSONB DEFAULT '{}'::jsonb,
                    relevant_observations JSONB DEFAULT '[]'::jsonb,
                    dependencies JSONB DEFAULT '[]'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    reviewed_at TIMESTAMP WITH TIME ZONE,
                    executed_at TIMESTAMP WITH TIME ZONE,
                    completed_at TIMESTAMP WITH TIME ZONE,
                    execution_result TEXT,
                    error_message TEXT
                );
            """)

            logger.info("Application tables created")

    async def _create_indexes(self) -> None:
        """Create performance indexes."""
        async with self.db_connection.get_connection() as conn:
            # Projects indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS projects_created_at_idx
                ON projects (created_at DESC);
            """)

            # Search history indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS search_history_project_idx
                ON search_history (project_id);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS search_history_created_at_idx
                ON search_history (created_at DESC);
            """)

            # Documents indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS documents_project_idx
                ON documents (project_id);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS documents_content_hash_idx
                ON documents (content_hash);
            """)

            # Analysis results indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS analysis_results_project_idx
                ON analysis_results (project_id);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS analysis_results_technique_idx
                ON analysis_results (technique_name);
            """)

            # Agent states indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS agent_states_project_idx
                ON agent_states (project_id);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS agent_states_role_idx
                ON agent_states (role_id);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS agent_states_active_idx
                ON agent_states (is_active);
            """)

            # Agent proposals indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS agent_proposals_project_idx
                ON agent_proposals (project_id);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS agent_proposals_agent_idx
                ON agent_proposals (agent_id);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS agent_proposals_status_idx
                ON agent_proposals (status);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS agent_proposals_priority_idx
                ON agent_proposals (priority);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS agent_proposals_created_at_idx
                ON agent_proposals (created_at DESC);
            """)

            logger.info("Database indexes created")

    async def health_check(self) -> DatabaseHealthStatus:
        """Perform comprehensive database health check."""
        start_time = datetime.now()
        try:
            async with self.db_connection.get_connection() as conn:
                # Check basic connectivity
                basic_check = await conn.fetchval("SELECT 1") == 1

                # Check pgvector extension
                vector_check = (
                    await conn.fetchval("""
                    SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'
                """)
                    > 0
                )

                # Check required tables exist
                table_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_name IN ('document_embeddings', 'projects', 'documents', 'agent_states', 'agent_proposals')
                """)
                tables_check = table_count >= 5

                # Test vector operations
                vector_ops_check = False
                try:
                    await conn.fetchval(
                        "SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector"
                    )
                    vector_ops_check = True
                except Exception:
                    pass

                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds() * 1000

                overall_health = (
                    basic_check and vector_check and tables_check and vector_ops_check
                )

                return DatabaseHealthStatus(
                    is_healthy=overall_health,
                    connection_status="connected" if basic_check else "disconnected",
                    table_count=table_count or 0,
                    last_check=end_time,
                    response_time_ms=response_time,
                )

        except Exception as e:
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000

            return DatabaseHealthStatus(
                is_healthy=False,
                connection_status="error",
                table_count=0,
                last_check=end_time,
                response_time_ms=response_time,
                error_message=str(e),
            )

    async def reset_database(self, confirm: bool = False) -> bool:
        """
        Reset database (DROP ALL DATA - use with extreme caution).

        Args:
            confirm: Must be True to proceed with reset

        Returns:
            True if reset successful, False otherwise
        """
        if not confirm:
            logger.warning("Database reset called without confirmation - ignoring")
            return False

        try:
            logger.warning("RESETTING DATABASE - ALL DATA WILL BE LOST!")

            async with self.db_connection.get_connection() as conn:
                # Drop all application tables
                await conn.execute("DROP TABLE IF EXISTS analysis_results CASCADE")
                await conn.execute("DROP TABLE IF EXISTS search_history CASCADE")
                await conn.execute("DROP TABLE IF EXISTS documents CASCADE")
                await conn.execute("DROP TABLE IF EXISTS document_embeddings CASCADE")
                await conn.execute("DROP TABLE IF EXISTS search_queries CASCADE")
                await conn.execute("DROP TABLE IF EXISTS projects CASCADE")

            # Reinitialize
            return await self.initialize_database()

        except Exception as e:
            logger.error(f"Database reset failed: {str(e)}")
            return False


async def main():
    """Main function for standalone database initialization."""
    logging.basicConfig(level=logging.INFO)

    # Initialize with default connection
    db_connection = PostgreSQLConnection()
    initializer = DatabaseInitializer(db_connection)

    # Initialize database
    success = await initializer.initialize_database()

    if success:
        # Perform health check
        health = await initializer.health_check()
        print("Database Health Check:")
        for key, value in health.items():
            status = "✅" if value else "❌"
            print(f"  {key}: {status}")

    await db_connection.close()
    return success


if __name__ == "__main__":
    asyncio.run(main())
