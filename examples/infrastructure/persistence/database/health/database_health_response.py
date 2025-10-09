"""Serializable database health response for API."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatabaseHealthResponse:
    """Serializable database health response for API."""

    is_healthy: bool
    connection_status: str
    table_count: int
    last_check: str  # ISO format string for JSON serialization
    response_time_ms: float
    error_message: str | None = None
