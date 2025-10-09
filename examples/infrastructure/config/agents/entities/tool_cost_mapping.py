"""Tool cost mapping entity."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCostMapping:
    """Mapping of tool name to cost."""

    tool_name: str
    cost: int
