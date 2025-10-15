from dataclasses import dataclass

from ml_lib.diffusion.prompt.attributes.attribute_config import AttributeConfig


@dataclass(frozen=True)
class ClothingConfig(AttributeConfig):
    """Configuration for clothing attributes."""

    pass
