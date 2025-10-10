"""Recommendations entity for assisted mode."""

from dataclasses import dataclass


@dataclass
class Recommendations:
    """AI recommendations for assisted mode."""

    prompt_analysis: "PromptAnalysis"
    """Analysis of the prompt."""

    suggested_loras: list["LoRARecommendation"]
    """Suggested LoRAs with confidence scores."""

    suggested_params: "OptimizedParameters"
    """Suggested generation parameters."""

    explanation: str
    """High-level explanation of recommendations."""

    def get_summary(self) -> str:
        """
        Get human-readable summary of recommendations.

        Returns:
            Formatted summary string
        """
        lines = [
            "=== AI Recommendations ===",
            "",
            f"Explanation: {self.explanation}",
            "",
            "Suggested LoRAs:",
        ]

        if self.suggested_loras:
            for rec in self.suggested_loras:
                lines.append(
                    f"  • {rec.lora_name} (confidence: {rec.confidence_score:.2f}, "
                    f"alpha: {rec.suggested_alpha:.2f})"
                )
        else:
            lines.append("  (none)")

        lines.extend([
            "",
            "Suggested Parameters:",
            f"  • Steps: {self.suggested_params.num_steps}",
            f"  • CFG Scale: {self.suggested_params.guidance_scale}",
            f"  • Resolution: {self.suggested_params.width}x{self.suggested_params.height}",
            f"  • Sampler: {self.suggested_params.sampler_name}",
            "",
        ])

        return "\n".join(lines)
