"""Single preset category entry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PresetCategoryEntry:
    """Single preset category with its preset names."""

    category_name: str
    preset_names: list[str]
