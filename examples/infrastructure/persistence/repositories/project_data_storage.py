"""Pure infrastructure storage using specific types."""

from infrastructure.persistence.repositories.project_storage_data import (
    ProjectStorageData,
)


class ProjectDataStorage:
    """Pure infrastructure storage using specific types."""

    def __init__(self):
        """Initialize empty storage."""
        self._projects: list[ProjectStorageData] = []

    def store(self, project_data: ProjectStorageData) -> None:
        """Store project data."""
        # Remove existing if present
        self._projects = [
            p for p in self._projects if p.project_id != project_data.project_id
        ]
        self._projects.append(project_data)

    def retrieve(self, project_id: str) -> ProjectStorageData | None:
        """Retrieve project data by ID."""
        for project in self._projects:
            if project.project_id == project_id:
                return project
        return None

    def retrieve_all(self) -> list[ProjectStorageData]:
        """Retrieve all project data."""
        return self._projects.copy()

    def remove(self, project_id: str) -> bool:
        """Remove project data."""
        original_count = len(self._projects)
        self._projects = [p for p in self._projects if p.project_id != project_id]
        return len(self._projects) < original_count

    def exists(self, project_id: str) -> bool:
        """Check if project exists."""
        return any(p.project_id == project_id for p in self._projects)
