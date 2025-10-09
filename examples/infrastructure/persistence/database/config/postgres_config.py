import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PostgresConfig:
    """Configuration for the PostgreSQL connection."""

    host: str = "localhost"
    port: int = 5432
    database: str = "pyintelcivil"
    user: str = "postgres"
    password: str = field(
        default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "postgres")
    )
    min_pool_size: int = 10
    max_pool_size: int = 20
