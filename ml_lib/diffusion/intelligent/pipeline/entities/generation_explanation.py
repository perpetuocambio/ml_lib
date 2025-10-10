"""Generation explanation entity."""

from dataclasses import dataclass, field


@dataclass
class GenerationExplanation:
    """Explanation of decisions made during generation."""

    summary: str
    """High-level summary of what was done."""

    lora_reasoning: dict[str, str] = field(default_factory=dict)
    """Reasoning for each LoRA selection (lora_name -> reasoning)."""

    parameter_reasoning: dict[str, str] = field(default_factory=dict)
    """Reasoning for parameter choices (param_name -> reasoning)."""

    performance_notes: list[str] = field(default_factory=list)
    """Performance-related notes (time, VRAM, etc.)."""

    def get_full_explanation(self) -> str:
        """
        Get full formatted explanation as multi-line string.

        Returns:
            Formatted explanation text
        """
        lines = [
            "=== Generation Explanation ===",
            "",
            "Summary:",
            f"  {self.summary}",
            "",
        ]

        if self.lora_reasoning:
            lines.append("LoRA Selection:")
            for lora_name, reasoning in self.lora_reasoning.items():
                lines.append(f"  • {lora_name}: {reasoning}")
            lines.append("")

        if self.parameter_reasoning:
            lines.append("Parameter Choices:")
            for param_name, reasoning in self.parameter_reasoning.items():
                lines.append(f"  • {param_name}: {reasoning}")
            lines.append("")

        if self.performance_notes:
            lines.append("Performance:")
            for note in self.performance_notes:
                lines.append(f"  • {note}")
            lines.append("")

        return "\n".join(lines)
