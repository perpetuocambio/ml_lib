"""Data structure for knowledge graph persistence."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeGraphData:
    """Typed data structure for knowledge graph database records."""

    graph_id: str
    project_id: str
    title: str
    description: str
    last_updated: str
