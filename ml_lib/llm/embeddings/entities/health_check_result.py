"""Health check result for embedding services."""

from dataclasses import dataclass


@dataclass
class EmbeddingHealthCheckResult:
    """Health check result for embedding services without using dictionaries."""

    is_healthy: bool
    model_loaded: bool
    can_generate_embeddings: bool
    expected_dimension: int
    actual_dimension: int | None
    error_message: str | None = None
    response_time_ms: float | None = None

    def has_dimension_mismatch(self) -> bool:
        """Check if there's a dimension mismatch."""
        if self.actual_dimension is None:
            return True
        return self.expected_dimension != self.actual_dimension

    def get_status_summary(self) -> str:
        """Get human-readable status summary."""
        if self.is_healthy:
            return f"✅ Embedding service healthy (dim: {self.actual_dimension})"

        issues = []
        if not self.model_loaded:
            issues.append("❌ Model not loaded")
        if not self.can_generate_embeddings:
            issues.append("❌ Cannot generate embeddings")
        if self.has_dimension_mismatch():
            issues.append(
                f"❌ Dimension mismatch: expected {self.expected_dimension}, got {self.actual_dimension}"
            )
        if self.error_message:
            issues.append(f"❌ Error: {self.error_message}")

        return "\n".join(issues)
