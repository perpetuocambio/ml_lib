"""Database health check status result."""

from dataclasses import dataclass
from datetime import datetime

from infrastructure.persistence.database.health.database_health_response import (
    DatabaseHealthResponse,
)


@dataclass(frozen=True)
class DatabaseHealthStatus:
    """Database health check status result."""

    is_healthy: bool
    connection_status: str
    table_count: int
    last_check: datetime
    response_time_ms: float
    error_message: str | None = None

    def to_response(self) -> DatabaseHealthResponse:
        """Convert to API response format."""
        return DatabaseHealthResponse(
            is_healthy=self.is_healthy,
            connection_status=self.connection_status,
            table_count=self.table_count,
            last_check=self.last_check.isoformat(),
            response_time_ms=self.response_time_ms,
            error_message=self.error_message,
        )
