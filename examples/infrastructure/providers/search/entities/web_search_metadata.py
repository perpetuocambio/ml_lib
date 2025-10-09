"""Metadata entity for web search results."""

from dataclasses import dataclass


@dataclass
class WebSearchMetadata:
    """Metadata for web search results without using dictionaries."""

    language: str
    region: str
    time_filter: str | None
    position: int
    provider_name: str
    search_duration_ms: float | None = None
    cache_hit: bool = False
    result_quality_score: float | None = None

    def to_display_text(self) -> str:
        """Convert metadata to human-readable display text."""
        parts = []

        parts.append(f"Position: {self.position}")
        parts.append(f"Provider: {self.provider_name}")
        parts.append(f"Language: {self.language}")

        if self.time_filter:
            parts.append(f"Time: {self.time_filter}")

        if self.search_duration_ms:
            parts.append(f"Duration: {self.search_duration_ms:.0f}ms")

        if self.cache_hit:
            parts.append("(Cached)")

        return " | ".join(parts)
