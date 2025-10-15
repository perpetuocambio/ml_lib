from dataclasses import dataclass, field

from ml_lib.diffusion.prompt.attributes.attribute_config import AttributeConfig


@dataclass(frozen=True)
class EthnicityConfig(AttributeConfig):
    """Configuration for ethnicity attributes."""

    prompt_weight: float = 1.0
    hair_colors: tuple[str, ...] = field(default_factory=tuple)
    hair_textures: tuple[str, ...] = field(default_factory=tuple)
    eye_colors: tuple[str, ...] = field(default_factory=tuple)
    skin_tones: tuple[str, ...] = field(default_factory=tuple)
