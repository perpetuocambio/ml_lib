"""PostgreSQL connection management for RAG and vector operations."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import asyncpg
from asyncpg import Pool
from infrastructure.persistence.database.config.postgres_config import PostgresConfig


class PostgreSQLConnection:
    """
    PostgreSQL connection manager with pgvector support.

    Manages connection pooling and provides vector search capabilities.
    """

    def __init__(self, config: PostgresConfig):
        """Initialize PostgreSQL connection with a config object."""
        self.config = config
        self._pool: Pool | None = None

    async def initialize_pool(self) -> None:
        """Initialize connection pool with pgvector extension."""
        if self._pool:
            return

        self._pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password,
            min_size=self.config.min_pool_size,
            max_size=self.config.max_pool_size,
            command_timeout=60,
        )

        # Ensure pgvector extension is enabled
        await self.ensure_pgvector_extension()

    async def ensure_pgvector_extension(self) -> None:
        """Ensure pgvector extension is enabled in the database."""
        async with self.get_connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get a connection from the pool."""
        if not self._pool:
            await self.initialize_pool()

        async with self._pool.acquire() as connection:
            yield connection

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def health_check(self) -> bool:
        """Perform health check on PostgreSQL connection."""
        try:
            async with self.get_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception:
            return False

    async def setup_vector_tables(self) -> None:
        """Setup vector tables for RAG operations."""
        async with self.get_connection() as conn:
            # Create documents table with vector embeddings
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id SERIAL PRIMARY KEY,
                    project_id VARCHAR(255) NOT NULL,
                    document_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(384),  # sentence-transformers/all-MiniLM-L6-v2 dimension
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)

            # Create index for vector similarity search
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS document_embeddings_vector_idx
                ON document_embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)

            # Create index for project filtering
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS document_embeddings_project_idx
                ON document_embeddings (project_id);
            """)

            # Create search queries table for caching
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS search_queries (
                    id SERIAL PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    query_embedding vector(384),
                    project_id VARCHAR(255),
                    results_cached JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)

            # Index for query caching
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS search_queries_vector_idx
                ON search_queries USING ivfflat (query_embedding vector_cosine_ops)
                WITH (lists = 100);
            """)

    @property
    def is_connected(self) -> bool:
        """Check if connection pool is active."""
        return self._pool is not None and not self._pool._closed
