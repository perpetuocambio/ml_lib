"""Error codes for MCP tool execution."""

from enum import Enum


class MCPToolErrorCode(Enum):
    """Enumerated error codes for MCP tool execution failures."""

    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    ARGUMENT_ERROR = "ARGUMENT_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
