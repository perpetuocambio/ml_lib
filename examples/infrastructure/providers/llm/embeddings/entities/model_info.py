"""Model information entity for embedding services."""

from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about an embedding model without using dictionaries."""

    model_name: str
    model_version: str
    embedding_dimension: int
    max_sequence_length: int
    model_provider: str
    supported_languages: list[str]
    is_multilingual: bool
    model_size_mb: float | None = None
    description: str = ""

    def get_display_name(self) -> str:
        """Get formatted display name."""
        return f"{self.model_name} v{self.model_version}"

    def supports_language(self, language_code: str) -> bool:
        """Check if model supports specific language."""
        return language_code.lower() in [
            lang.lower() for lang in self.supported_languages
        ]
