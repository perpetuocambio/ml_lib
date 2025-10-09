"""Database migration and initialization script."""

import asyncio
import logging
from pathlib import Path

from infrastructure.persistence.database.config.database_config import (
    DatabaseConfigFactory,
)
from infrastructure.persistence.database.config.database_connection_factory import (
    DatabaseConnectionFactory,
)

logger = logging.getLogger(__name__)


class DatabaseMigrationScript:
    """Database migration and initialization utility."""

    def __init__(self):
        """Initialize migration script."""
        self.config = DatabaseConfigFactory.from_environment()
        self.connection = DatabaseConnectionFactory.create_connection(self.config)

    async def initialize_database(self) -> bool:
        """Initialize complete database schema."""
        try:
            logger.info(f"Initializing {self.config.database_type.value} database...")

            # Initialize connection pool
            await self.connection.initialize_pool()

            # Initialize complete schema
            await self.connection.initialize_schema()

            # Setup vector tables for RAG
            await self.connection.setup_vector_tables()

            logger.info("Database initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            return False
        finally:
            await self.connection.close()

    async def health_check(self) -> bool:
        """Perform database health check."""
        try:
            logger.info("Performing database health check...")

            health_result = await self.connection.health_check()

            print("Database Health Check:")
            print(health_result.get_status_summary())

            return health_result.is_healthy()

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
        finally:
            await self.connection.close()

    async def migrate_from_sqlite(self, sqlite_db_path: Path) -> bool:
        """Migrate data from SQLite to PostgreSQL (if using PostgreSQL)."""
        if self.config.database_type.value != "postgresql":
            logger.info("Not using PostgreSQL, skipping migration")
            return True

        try:
            logger.info("Starting migration from SQLite to PostgreSQL...")

            # TODO: Implement actual data migration logic
            # This would involve:
            # 1. Reading data from SQLite files
            # 2. Transforming data format if needed
            # 3. Inserting data into PostgreSQL

            logger.warning("Migration not yet implemented - manual migration required")
            return True

        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            return False


async def main() -> None:
    """Main migration script entry point."""
    logging.basicConfig(level=logging.INFO)

    migration_script = DatabaseMigrationScript()

    # Initialize database
    init_success = await migration_script.initialize_database()

    if init_success:
        # Perform health check
        health_success = await migration_script.health_check()

        if health_success:
            print("✅ Database is ready for use!")
            return

    print("❌ Database initialization failed")


if __name__ == "__main__":
    asyncio.run(main())
