"""Database infrastructure module."""

from infrastructure.persistence.database.config import (
    DatabaseConfig,
    DatabaseConfigFactory,
)
from infrastructure.persistence.database.connections.database_connection import (
    IDatabaseConnection,
)
from infrastructure.persistence.database.enums.database_type import DatabaseType
from infrastructure.persistence.database.health.database_health_response import (
    DatabaseHealthResponse,
)
from infrastructure.persistence.database.health.database_health_status_class import (
    DatabaseHealthStatus,
)
from infrastructure.persistence.database.queries.query_parameters import QueryParameters
from infrastructure.persistence.database.queries.query_result import QueryResult

# DatabaseMigrationScript temporarily disabled due to missing dependencies

__all__ = [
    "DatabaseConfig",
    "DatabaseConfigFactory",
    "IDatabaseConnection",
    "DatabaseType",
    "DatabaseHealthResponse",
    "DatabaseHealthStatus",
    "QueryParameters",
    "QueryResult",
]
