"""Execution result data for WebSocket messages."""

from pydantic import BaseModel


class ExecutionResultData(BaseModel):
    """Type-safe container for execution result data."""

    success: bool = False
    output_message: str = ""
    artifacts_created: str = ""
    duration_seconds: str = "0"
    tool_used: str = ""
    error_code: str = ""
