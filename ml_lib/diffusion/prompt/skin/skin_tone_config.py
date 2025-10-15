from dataclasses import dataclass, field

from ml_lib.diffusion.prompt.common.attribute import AttributeConfig


@dataclass(frozen=True)
class SkinToneConfig(AttributeConfig):
    """Configuration for skin tone attributes."""

    prompt_weight: float = 1.1
    ethnicity_associations: tuple[str, ...] = field(default_factory=tuple)
