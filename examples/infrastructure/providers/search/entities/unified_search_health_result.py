"""Health check result for unified search service."""

from dataclasses import dataclass

from infrastructure.providers.search.entities.semantic_search_health_check_result import (
    SemanticSearchHealthCheckResult,
)


@dataclass
class UnifiedSearchHealthResult:
    """Health check result for unified search service without using dictionaries."""

    web_search_healthy: bool
    semantic_search_health: SemanticSearchHealthCheckResult | None
    overall_healthy: bool
    web_provider_name: str
    error_message: str | None = None

    def get_status_summary(self) -> str:
        """Get comprehensive health status summary."""
        if self.overall_healthy:
            components = [self.web_provider_name]
            if self.semantic_search_health:
                components.append("Semantic Search")
            return f"✅ Unified search fully operational ({' + '.join(components)})"

        issues = []

        if not self.web_search_healthy:
            issues.append(f"🌐 Web search ({self.web_provider_name}) not available")

        if (
            self.semantic_search_health
            and not self.semantic_search_health.overall_healthy
        ):
            issues.append("🔍 Semantic search issues:")
            issues.append(self.semantic_search_health.get_status_summary())

        if self.error_message:
            issues.append(f"❌ System Error: {self.error_message}")

        return "\n".join(issues)
