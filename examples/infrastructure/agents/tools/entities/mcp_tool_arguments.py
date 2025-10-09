"""MCP tool arguments domain entity."""

from dataclasses import dataclass, field

from infrastructure.agents.tools.entities.parameter_entry import ParameterEntry


@dataclass(frozen=True)
class MCPToolArguments:
    """Typed arguments for MCP tool execution."""

    tool_name: str
    parameters: list[ParameterEntry] = field(default_factory=list)

    def get_parameter_value(self, key: str) -> str | None:
        """Get parameter value by key."""
        for param in self.parameters:
            if param.key == key:
                return param.value
        return None

    def has_parameter(self, key: str) -> bool:
        """Check if parameter exists."""
        return any(param.key == key for param in self.parameters)
