"""Infrastructure-specific project data structure."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ProjectStorageData:
    """Infrastructure-specific project data structure."""

    project_id: str
    name: str
    description: str
    status: str
    project_type: str
    created_at: datetime
    updated_at: datetime
    language: str
    region: str
    discipline: str
