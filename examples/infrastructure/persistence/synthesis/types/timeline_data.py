"""Data structure for timeline persistence."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TimelineData:
    """Typed data structure for timeline database records."""

    timeline_id: str
    project_id: str
    title: str
    description: str
    last_updated: str
