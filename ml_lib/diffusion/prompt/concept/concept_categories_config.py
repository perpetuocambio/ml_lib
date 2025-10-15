from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConceptCategoriesConfig:
    """Concept categories for prompt analysis."""

    character: tuple[str, ...]
    style: tuple[str, ...]
    content: tuple[str, ...]
    setting: tuple[str, ...]
    quality: tuple[str, ...]
    lighting: tuple[str, ...]
    camera: tuple[str, ...]
    technical: tuple[str, ...]
    subjects: tuple[str, ...]
    age_attributes: tuple[str, ...]
    relationships: tuple[str, ...]
    youth_indicators: tuple[str, ...]
    adult_indicators: tuple[str, ...]
    medical_conditions: tuple[str, ...]
    anatomy: tuple[str, ...]
    physical_details: tuple[str, ...]
    activity: tuple[str, ...]
    clothing: tuple[str, ...]
