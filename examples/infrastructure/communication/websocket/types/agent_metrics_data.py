"""Agent metrics data for WebSocket messages."""

from pydantic import BaseModel


class AgentMetricsData(BaseModel):
    """Type-safe container for agent metrics data."""

    tasks_completed: int = 0
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    memory_usage: str = "0MB"
    cpu_usage: str = "0%"
    uptime_minutes: int = 0
    errors_count: int = 0
