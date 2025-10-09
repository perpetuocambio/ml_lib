"""Type-safe preset catalog - replaces dict usage."""

from dataclasses import dataclass

from infrastructure.config.templates.preset_category_entry import PresetCategoryEntry


@dataclass(frozen=True)
class PresetCatalog:
    """Type-safe container for configuration presets - replaces dict with typed classes."""

    entries: list[PresetCategoryEntry]

    @classmethod
    def create_default(cls) -> "PresetCatalog":
        """Create default preset catalog."""
        return cls(
            entries=[
                PresetCategoryEntry(
                    category_name="llm",
                    preset_names=[
                        "development_ollama",
                        "production_openai",
                        "production_anthropic",
                        "fast_inference",
                    ],
                ),
                PresetCategoryEntry(
                    category_name="scraping",
                    preset_names=["respectful", "fast", "comprehensive", "research"],
                ),
                PresetCategoryEntry(
                    category_name="extraction",
                    preset_names=["fast", "comprehensive", "text_only", "academic"],
                ),
            ]
        )

    def get_presets_for_category(self, category_name: str) -> list[str]:
        """Get preset names for a specific category."""
        for entry in self.entries:
            if entry.category_name == category_name:
                return entry.preset_names.copy()
        return []

    def get_all_category_names(self) -> list[str]:
        """Get all available category names."""
        return [entry.category_name for entry in self.entries]

    def has_category(self, category_name: str) -> bool:
        """Check if category exists."""
        return any(entry.category_name == category_name for entry in self.entries)
