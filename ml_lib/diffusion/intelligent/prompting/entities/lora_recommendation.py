"""LoRA recommendation entities."""

from dataclasses import dataclass, field

from ml_lib.diffusion.intelligent.hub_integration.entities import ModelMetadata


@dataclass
class LoRARecommendation:
    """Recommendation for a LoRA."""

    lora_name: str
    lora_metadata: ModelMetadata
    confidence_score: float
    suggested_alpha: float
    matching_concepts: list[str] = field(default_factory=list)
    reasoning: str = ""

    def __post_init__(self):
        """Validate recommendation."""
        assert 0.0 <= self.confidence_score <= 1.0, "Confidence must be between 0 and 1"
        assert 0.0 < self.suggested_alpha <= 2.0, "Alpha should be between 0 and 2"

    def is_compatible_with(self, other: "LoRARecommendation") -> bool:
        """
        Check compatibility with another LoRA.

        Args:
            other: Another LoRA recommendation

        Returns:
            True if compatible
        """
        # Check for style conflicts
        style_keywords = ["anime", "photorealistic", "cartoon", "3d", "realistic"]

        self_styles = [kw for kw in style_keywords if kw in self.lora_name.lower()]
        other_styles = [kw for kw in style_keywords if kw in other.lora_name.lower()]

        # If both have conflicting styles, not compatible
        if self_styles and other_styles:
            if set(self_styles).isdisjoint(set(other_styles)):
                return False

        # Check base model compatibility
        if self.lora_metadata.base_model != other.lora_metadata.base_model:
            return False

        return True
