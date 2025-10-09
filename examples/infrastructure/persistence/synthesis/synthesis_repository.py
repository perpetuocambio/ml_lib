"""
Repository for synthesis engine data persistence.

EPIC-003: Manages knowledge graphs, timelines, and extracted entities.
Infrastructure layer - works with primitives only.
"""

from infrastructure.persistence.base_repository import BaseRepository
from infrastructure.persistence.synthesis.types.entity_relationship_data import (
    EntityRelationshipData,
)
from infrastructure.persistence.synthesis.types.extracted_entity_data import (
    ExtractedEntityData,
)
from infrastructure.persistence.synthesis.types.knowledge_graph_data import (
    KnowledgeGraphData,
)
from infrastructure.persistence.synthesis.types.timeline_data import TimelineData
from infrastructure.persistence.synthesis.types.timeline_event_data import (
    TimelineEventData,
)
from infrastructure.serialization.protocol_serializer import ProtocolSerializer


class SynthesisRepository(BaseRepository):
    """
    Repository for synthesis data persistence.

    Works with primitive data types passed from Application layer.
    Does not import Domain entities to maintain architectural independence.
    """

    def __init__(self, db_path: str = "pyintelcivil.db") -> None:
        """Initialize synthesis repository."""
        super().__init__(db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        """Create synthesis tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Extracted entities table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extracted_entities (
                    entity_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    source_agent_id TEXT NOT NULL,
                    source_execution_id TEXT NOT NULL,
                    extraction_timestamp TEXT NOT NULL,
                    attributes_json TEXT,
                    related_entities_json TEXT,
                    geographic_info_json TEXT,
                    temporal_info_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Knowledge graphs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_graphs (
                    graph_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    title TEXT,
                    description TEXT,
                    last_updated TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Entity relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entity_relationships (
                    relationship_id TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    source_entity_id TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence TEXT NOT NULL,
                    source_agent_id TEXT NOT NULL,
                    created_timestamp TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (graph_id) REFERENCES knowledge_graphs (graph_id)
                )
            """)

            # Synthesis timelines table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS synthesis_timelines (
                    timeline_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Timeline events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS timeline_events (
                    event_id TEXT PRIMARY KEY,
                    timeline_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_agent_id TEXT NOT NULL,
                    source_execution_id TEXT NOT NULL,
                    related_entities_json TEXT,
                    end_timestamp TEXT,
                    duration_description TEXT,
                    evidence TEXT,
                    significance_score REAL DEFAULT 0.5,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (timeline_id) REFERENCES synthesis_timelines (timeline_id)
                )
            """)

            # Create indexes for better performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_project ON extracted_entities(source_agent_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_type ON extracted_entities(entity_type)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_graphs_project ON knowledge_graphs(project_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_relationships_source ON entity_relationships(source_entity_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_relationships_target ON entity_relationships(target_entity_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_timelines_project ON synthesis_timelines(project_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_timeline ON timeline_events(timeline_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON timeline_events(timestamp)"
            )

            conn.commit()

    # Extracted Entities Operations
    def save_extracted_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        description: str,
        confidence: str,
        source_agent_id: str,
        source_execution_id: str,
        extraction_timestamp: str,
        attributes_data: list[tuple[str, str]],
        related_entities: list[str],
        geographic_latitude: float | None = None,
        geographic_longitude: float | None = None,
        geographic_location_name: str | None = None,
        geographic_country: str | None = None,
        geographic_region: str | None = None,
        temporal_start_date: str | None = None,
        temporal_end_date: str | None = None,
        temporal_duration_description: str | None = None,
    ) -> None:
        """Save extracted entity with primitive data types."""
        # Serialize complex data using ProtocolSerializer
        attributes_json = ProtocolSerializer.serialize_mapping_data(
            dict(attributes_data)
        )
        related_entities_json = ProtocolSerializer.serialize_dict_data(
            {"entities": related_entities}
        )

        geographic_info_json = None
        if geographic_latitude is not None and geographic_longitude is not None:
            geographic_data = {
                "latitude": geographic_latitude,
                "longitude": geographic_longitude,
                "location_name": geographic_location_name,
                "country": geographic_country,
                "region": geographic_region,
            }
            geographic_info_json = ProtocolSerializer.serialize_dict_data(
                geographic_data
            )

        temporal_info_json = None
        if temporal_start_date is not None:
            temporal_data = {
                "start_date": temporal_start_date,
                "end_date": temporal_end_date,
                "duration_description": temporal_duration_description,
            }
            temporal_info_json = ProtocolSerializer.serialize_dict_data(temporal_data)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO extracted_entities
                (entity_id, name, entity_type, description, confidence, source_agent_id,
                 source_execution_id, extraction_timestamp, attributes_json, related_entities_json,
                 geographic_info_json, temporal_info_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (
                    entity_id,
                    name,
                    entity_type,
                    description,
                    confidence,
                    source_agent_id,
                    source_execution_id,
                    extraction_timestamp,
                    str(attributes_json),
                    str(related_entities_json),
                    str(geographic_info_json) if geographic_info_json else None,
                    str(temporal_info_json) if temporal_info_json else None,
                ),
            )
            conn.commit()

    def get_entities_by_project_agent(self, agent_id: str) -> list[ExtractedEntityData]:
        """Get all entities by agent ID as typed data."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT entity_id, name, entity_type, description, confidence, source_agent_id,
                       source_execution_id, extraction_timestamp, attributes_json, related_entities_json,
                       geographic_info_json, temporal_info_json
                FROM extracted_entities
                WHERE source_agent_id = ?
                ORDER BY extraction_timestamp DESC
            """,
                (agent_id,),
            )

            entities = []
            for row in cursor.fetchall():
                entity_data = ExtractedEntityData(
                    entity_id=row[0],
                    name=row[1],
                    entity_type=row[2],
                    description=row[3],
                    confidence=row[4],
                    source_agent_id=row[5],
                    source_execution_id=row[6],
                    extraction_timestamp=row[7],
                    attributes_json=row[8],
                    related_entities_json=row[9],
                    geographic_info_json=row[10],
                    temporal_info_json=row[11],
                )
                entities.append(entity_data)
            return entities

    def get_entity_by_id(self, entity_id: str) -> ExtractedEntityData | None:
        """Get entity by ID as typed data."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT entity_id, name, entity_type, description, confidence, source_agent_id,
                       source_execution_id, extraction_timestamp, attributes_json, related_entities_json,
                       geographic_info_json, temporal_info_json
                FROM extracted_entities
                WHERE entity_id = ?
            """,
                (entity_id,),
            )

            row = cursor.fetchone()
            if row:
                return ExtractedEntityData(
                    entity_id=row[0],
                    name=row[1],
                    entity_type=row[2],
                    description=row[3],
                    confidence=row[4],
                    source_agent_id=row[5],
                    source_execution_id=row[6],
                    extraction_timestamp=row[7],
                    attributes_json=row[8],
                    related_entities_json=row[9],
                    geographic_info_json=row[10],
                    temporal_info_json=row[11],
                )
            return None

    # Knowledge Graph Operations
    def save_knowledge_graph(
        self,
        graph_id: str,
        project_id: str,
        last_updated: str,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        """Save knowledge graph metadata."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO knowledge_graphs
                (graph_id, project_id, title, description, last_updated, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (graph_id, project_id, title, description, last_updated),
            )
            conn.commit()

    def save_entity_relationship(
        self,
        relationship_id: str,
        graph_id: str,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: str,
        confidence: float,
        evidence: str,
        source_agent_id: str,
        created_timestamp: str,
    ) -> None:
        """Save entity relationship."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO entity_relationships
                (relationship_id, graph_id, source_entity_id, target_entity_id, relationship_type,
                 confidence, evidence, source_agent_id, created_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    relationship_id,
                    graph_id,
                    source_entity_id,
                    target_entity_id,
                    relationship_type,
                    confidence,
                    evidence,
                    source_agent_id,
                    created_timestamp,
                ),
            )
            conn.commit()

    def get_relationships_by_graph(self, graph_id: str) -> list[EntityRelationshipData]:
        """Get all relationships for a knowledge graph."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT relationship_id, source_entity_id, target_entity_id, relationship_type,
                       confidence, evidence, source_agent_id, created_timestamp
                FROM entity_relationships
                WHERE graph_id = ?
                ORDER BY created_timestamp DESC
            """,
                (graph_id,),
            )

            relationships = []
            for row in cursor.fetchall():
                relationship_data = EntityRelationshipData(
                    relationship_id=row[0],
                    source_entity_id=row[1],
                    target_entity_id=row[2],
                    relationship_type=row[3],
                    confidence=row[4],
                    evidence=row[5],
                    source_agent_id=row[6],
                    created_timestamp=row[7],
                )
                relationships.append(relationship_data)
            return relationships

    # Timeline Operations
    def save_synthesis_timeline(
        self,
        timeline_id: str,
        project_id: str,
        title: str,
        description: str,
        last_updated: str,
    ) -> None:
        """Save synthesis timeline metadata."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO synthesis_timelines
                (timeline_id, project_id, title, description, last_updated, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (timeline_id, project_id, title, description, last_updated),
            )
            conn.commit()

    def save_timeline_event(
        self,
        event_id: str,
        timeline_id: str,
        title: str,
        description: str,
        event_type: str,
        timestamp: str,
        confidence: float,
        source_agent_id: str,
        source_execution_id: str,
        related_entities: list[str],
        end_timestamp: str | None = None,
        duration_description: str | None = None,
        evidence: str | None = None,
        significance_score: float = 0.5,
    ) -> None:
        """Save timeline event."""
        related_entities_json = ProtocolSerializer.serialize_dict_data(
            {"entities": related_entities}
        )

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO timeline_events
                (event_id, timeline_id, title, description, event_type, timestamp, confidence,
                 source_agent_id, source_execution_id, related_entities_json, end_timestamp,
                 duration_description, evidence, significance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event_id,
                    timeline_id,
                    title,
                    description,
                    event_type,
                    timestamp,
                    confidence,
                    source_agent_id,
                    source_execution_id,
                    str(related_entities_json),
                    end_timestamp,
                    duration_description,
                    evidence,
                    significance_score,
                ),
            )
            conn.commit()

    def get_events_by_timeline(self, timeline_id: str) -> list[TimelineEventData]:
        """Get all events for a timeline."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT event_id, title, description, event_type, timestamp, confidence,
                       source_agent_id, source_execution_id, related_entities_json, end_timestamp,
                       duration_description, evidence, significance_score
                FROM timeline_events
                WHERE timeline_id = ?
                ORDER BY timestamp ASC
            """,
                (timeline_id,),
            )

            events = []
            for row in cursor.fetchall():
                event_data = TimelineEventData(
                    event_id=row[0],
                    title=row[1],
                    description=row[2],
                    event_type=row[3],
                    timestamp=row[4],
                    confidence=row[5],
                    source_agent_id=row[6],
                    source_execution_id=row[7],
                    related_entities_json=row[8],
                    end_timestamp=row[9],
                    duration_description=row[10],
                    evidence=row[11],
                    significance_score=row[12],
                )
                events.append(event_data)
            return events

    def get_timeline_by_project(self, project_id: str) -> TimelineData | None:
        """Get timeline by project ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT timeline_id, title, description, last_updated
                FROM synthesis_timelines
                WHERE project_id = ?
            """,
                (project_id,),
            )

            row = cursor.fetchone()
            if row:
                return TimelineData(
                    timeline_id=row[0],
                    project_id=project_id,
                    title=row[1],
                    description=row[2],
                    last_updated=row[3],
                )
            return None

    def get_knowledge_graph_by_project(
        self, project_id: str
    ) -> KnowledgeGraphData | None:
        """Get knowledge graph by project ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT graph_id, title, description, last_updated
                FROM knowledge_graphs
                WHERE project_id = ?
            """,
                (project_id,),
            )

            row = cursor.fetchone()
            if row:
                return KnowledgeGraphData(
                    graph_id=row[0],
                    project_id=project_id,
                    title=row[1],
                    description=row[2],
                    last_updated=row[3],
                )
            return None
