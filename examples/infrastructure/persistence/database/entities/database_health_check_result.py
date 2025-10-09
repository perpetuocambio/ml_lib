"""Database health check result entity."""

from dataclasses import dataclass


@dataclass
class DatabaseHealthCheckResult:
    """Database health check result without using dictionaries."""

    basic_connectivity: bool
    pgvector_extension: bool
    required_tables: bool
    vector_operations: bool
    overall_health: bool
    error_message: str | None = None

    def is_healthy(self) -> bool:
        """Check if database is healthy."""
        return self.overall_health

    def get_status_summary(self) -> str:
        """Get human-readable status summary."""
        if self.overall_health:
            return "✅ Database is healthy and operational"

        issues = []
        if not self.basic_connectivity:
            issues.append("❌ Database connection failed")
        if not self.pgvector_extension:
            issues.append("❌ pgvector extension not available")
        if not self.required_tables:
            issues.append("❌ Required tables missing")
        if not self.vector_operations:
            issues.append("❌ Vector operations not working")

        if self.error_message:
            issues.append(f"❌ Error: {self.error_message}")

        return "\n".join(issues)
