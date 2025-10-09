"""Persistence infrastructure module - handles data storage and retrieval."""

# Database access
from infrastructure.persistence.database import (
    DatabaseConfig,
    DatabaseConfigFactory,
    DatabaseHealthResponse,
    DatabaseHealthStatus,
    DatabaseType,
    IDatabaseConnection,
    QueryParameters,
    QueryResult,
)

# Storage access
from infrastructure.persistence.storage import (
    LocalStorageService,
    StorageInterface,
)

__all__ = [
    # Database
    "DatabaseConfig",
    "DatabaseConfigFactory",
    "IDatabaseConnection",
    "DatabaseType",
    "DatabaseHealthResponse",
    "DatabaseHealthStatus",
    "QueryParameters",
    "QueryResult",
    # Storage
    "StorageInterface",
    "LocalStorageService",
]
