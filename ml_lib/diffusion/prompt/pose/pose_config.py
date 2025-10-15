from dataclasses import dataclass
from typing_extensions import Literal
from ml_lib.diffusion.prompt.attributes.attribute_config import AttributeConfig


@dataclass(frozen=True)
class PoseConfig(AttributeConfig):
    """Configuration for pose attributes."""

    complexity: Literal["low", "medium", "high"] = "low"
    explicit: bool = False
