from dataclasses import dataclass, field

from ml_lib.diffusion.prompt.attributes.attribute_config import AttributeConfig


@dataclass(frozen=True)
class AgeRangeConfig(AttributeConfig):
    """Configuration for age range attributes."""

    age_min: int = 18
    age_max: int = 80
    age_features: tuple[str, ...] = field(default_factory=tuple)
