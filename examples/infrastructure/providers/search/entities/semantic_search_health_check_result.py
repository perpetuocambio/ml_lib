"""Health check result for semantic search service."""

from dataclasses import dataclass

from infrastructure.persistence.database.database_health_check_result import (
    DatabaseHealthCheckResult,
)
from infrastructure.providers.llm.embeddings.entities.health_check_result import (
    EmbeddingHealthCheckResult,
)


@dataclass
class SemanticSearchHealthCheckResult:
    """Health check result for semantic search service without using dictionaries."""

    database_health: DatabaseHealthCheckResult
    embedding_health: EmbeddingHealthCheckResult
    vector_operations_working: bool
    overall_healthy: bool
    error_message: str | None = None

    def get_status_summary(self) -> str:
        """Get comprehensive health status summary."""
        if self.overall_healthy:
            return "✅ Semantic search system fully operational"

        issues = []

        if not self.database_health.is_healthy():
            issues.append("📊 Database Issues:")
            issues.append(self.database_health.get_status_summary())

        if not self.embedding_health.is_healthy:
            issues.append("🧠 Embedding Issues:")
            issues.append(self.embedding_health.get_status_summary())

        if not self.vector_operations_working:
            issues.append("🔍 Vector operations not working")

        if self.error_message:
            issues.append(f"❌ System Error: {self.error_message}")

        return "\n".join(issues)
