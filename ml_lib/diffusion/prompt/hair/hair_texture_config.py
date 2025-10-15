from dataclasses import dataclass, field

from ml_lib.diffusion.prompt.attributes.attribute_config import AttributeConfig


@dataclass(frozen=True)
class HairTextureConfig(AttributeConfig):
    """Configuration for hair texture attributes."""

    ethnicity_fit: tuple[str, ...] = field(default_factory=tuple)
    prompt_weight: float = 1.0
