"""Decision explainer for explaining AI decisions in generation."""

import logging
from typing import Optional, Protocol
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PromptAnalysisProtocol(Protocol):
    """Protocol for prompt analysis results."""

    pass


class GenerationResultProtocol(Protocol):
    """Protocol for generation results."""

    pass


class ExplanationVerbosity(Enum):
    """Verbosity level for explanations."""

    MINIMAL = "minimal"
    """Only essential information."""

    STANDARD = "standard"
    """Standard level of detail."""

    DETAILED = "detailed"
    """Comprehensive explanations."""

    TECHNICAL = "technical"
    """Include technical details and reasoning."""


@dataclass
class DecisionContext:
    """Context for a decision that was made."""

    decision_type: str
    """Type of decision (e.g., 'lora_selection', 'parameter_choice')."""

    decision_value: str
    """The actual decision/value chosen."""

    alternatives: list[str]
    """Other options that were considered."""

    reasoning: str
    """Why this decision was made."""

    confidence: float
    """Confidence in this decision (0-1)."""

    data_sources: list[str]
    """What data informed this decision."""


class DecisionExplainer:
    """
    Explains decisions made by the intelligent pipeline.

    Provides human-readable explanations for:
    - Why specific LoRAs were chosen
    - Why certain parameters were set
    - What trade-offs were made
    - How the system analyzed the prompt

    Example:
        >>> explainer = DecisionExplainer()
        >>> explanation = explainer.explain_lora_selection(
        ...     lora_name="anime_style_v2",
        ...     confidence=0.85,
        ...     reasoning="Matched 'anime' keyword with high semantic similarity"
        ... )
        >>> print(explanation)
    """

    def __init__(self, verbosity: ExplanationVerbosity = ExplanationVerbosity.STANDARD):
        """
        Initialize decision explainer.

        Args:
            verbosity: Explanation verbosity level
        """
        self.verbosity = verbosity

    def explain_lora_selection(
        self,
        lora_name: str,
        confidence: float,
        reasoning: str,
        alternative_loras: Optional[list[tuple[str, float]]] = None,
    ) -> str:
        """
        Explain why a LoRA was selected.

        Args:
            lora_name: Name of selected LoRA
            confidence: Confidence score (0-1)
            reasoning: Reasoning text
            alternative_loras: List of (name, score) for alternatives

        Returns:
            Explanation text
        """
        explanation_parts = []

        # Basic explanation
        explanation_parts.append(f"Selected LoRA '{lora_name}' (confidence: {confidence:.2f})")
        explanation_parts.append(f"Reason: {reasoning}")

        # Add alternatives if verbose
        if self.verbosity in (ExplanationVerbosity.DETAILED, ExplanationVerbosity.TECHNICAL):
            if alternative_loras:
                explanation_parts.append("\nAlternatives considered:")
                for alt_name, alt_score in alternative_loras[:3]:
                    explanation_parts.append(f"  - {alt_name}: {alt_score:.2f}")

        # Add technical details if requested
        if self.verbosity == ExplanationVerbosity.TECHNICAL:
            explanation_parts.append(
                f"\nTechnical: Confidence threshold met (>0.6), "
                f"semantic match confirmed"
            )

        return "\n".join(explanation_parts)

    def explain_parameter_choice(
        self,
        param_name: str,
        param_value: str,
        reasoning: str,
        default_value: Optional[str] = None,
        constraints: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Explain why a parameter was set to a specific value.

        Args:
            param_name: Parameter name
            param_value: Chosen value (as string for display)
            reasoning: Reasoning text
            default_value: Default value (if different)
            constraints: Any constraints that influenced the choice

        Returns:
            Explanation text

        Note:
            All values should be passed as strings for display purposes.
            Callers should convert numeric/other values to strings before calling.
        """
        explanation_parts = []

        # Basic explanation
        explanation_parts.append(f"{param_name} = {param_value}")
        explanation_parts.append(f"Reason: {reasoning}")

        # Show if different from default
        if default_value is not None and default_value != param_value:
            explanation_parts.append(f"(default would be: {default_value})")

        # Add constraints if verbose
        if self.verbosity in (ExplanationVerbosity.DETAILED, ExplanationVerbosity.TECHNICAL):
            if constraints:
                explanation_parts.append("\nConstraints applied:")
                for constraint_name, constraint_value in constraints.items():
                    explanation_parts.append(f"  - {constraint_name}: {constraint_value}")

        return "\n".join(explanation_parts)

    def explain_trade_off(
        self,
        choice_made: str,
        benefits: list[str],
        costs: list[str],
    ) -> str:
        """
        Explain a trade-off decision.

        Args:
            choice_made: The choice that was made
            benefits: Benefits of this choice
            costs: Costs/downsides of this choice

        Returns:
            Explanation text
        """
        explanation_parts = []

        explanation_parts.append(f"Trade-off: {choice_made}")

        if self.verbosity != ExplanationVerbosity.MINIMAL:
            explanation_parts.append("\nBenefits:")
            for benefit in benefits:
                explanation_parts.append(f"  + {benefit}")

            if costs:
                explanation_parts.append("\nTrade-offs:")
                for cost in costs:
                    explanation_parts.append(f"  - {cost}")

        return "\n".join(explanation_parts)

    def explain_prompt_analysis(
        self,
        prompt: str,
        identified_concepts: list[str],
        identified_style: str,
        complexity_level: str,
        special_requirements: Optional[list[str]] = None,
    ) -> str:
        """
        Explain how the prompt was analyzed.

        Args:
            prompt: Original prompt
            identified_concepts: Key concepts identified
            identified_style: Artistic style detected
            complexity_level: Complexity assessment
            special_requirements: Special requirements detected

        Returns:
            Explanation text
        """
        explanation_parts = []

        explanation_parts.append(f"Prompt Analysis: \"{prompt[:50]}...\"")
        explanation_parts.append(f"\nStyle: {identified_style}")
        explanation_parts.append(f"Complexity: {complexity_level}")

        if self.verbosity != ExplanationVerbosity.MINIMAL:
            explanation_parts.append("\nKey Concepts:")
            for concept in identified_concepts[:5]:
                explanation_parts.append(f"  • {concept}")

            if special_requirements:
                explanation_parts.append("\nSpecial Requirements:")
                for req in special_requirements:
                    explanation_parts.append(f"  • {req}")

        return "\n".join(explanation_parts)

    def explain_complete_decision_chain(
        self,
        prompt: str,
        analysis: PromptAnalysisProtocol,
        lora_decisions: list[DecisionContext],
        param_decisions: list[DecisionContext],
        final_outcome: str,
    ) -> str:
        """
        Explain the complete decision chain from prompt to outcome.

        Args:
            prompt: Original prompt
            analysis: Prompt analysis results
            lora_decisions: LoRA selection decisions
            param_decisions: Parameter optimization decisions
            final_outcome: Summary of final outcome

        Returns:
            Complete explanation text
        """
        explanation_parts = []

        explanation_parts.append("=== Complete Decision Chain ===\n")

        # 1. Prompt Analysis
        explanation_parts.append("1. Prompt Analysis")
        if hasattr(analysis, "identified_concepts"):
            explanation_parts.append(
                f"   Identified {len(analysis.identified_concepts)} key concepts"
            )
        if hasattr(analysis, "intent"):
            explanation_parts.append(f"   Style: {analysis.intent.artistic_style.value}")
        explanation_parts.append("")

        # 2. LoRA Selection
        explanation_parts.append("2. LoRA Selection")
        if lora_decisions:
            for i, decision in enumerate(lora_decisions, 1):
                explanation_parts.append(
                    f"   {i}. {decision.decision_value} "
                    f"(confidence: {decision.confidence:.2f})"
                )
                if self.verbosity != ExplanationVerbosity.MINIMAL:
                    explanation_parts.append(f"      Reason: {decision.reasoning}")
        else:
            explanation_parts.append("   (none)")
        explanation_parts.append("")

        # 3. Parameter Optimization
        explanation_parts.append("3. Parameter Optimization")
        for decision in param_decisions:
            explanation_parts.append(
                f"   • {decision.decision_type}: {decision.decision_value}"
            )
            if self.verbosity != ExplanationVerbosity.MINIMAL:
                explanation_parts.append(f"      {decision.reasoning}")
        explanation_parts.append("")

        # 4. Final Outcome
        explanation_parts.append("4. Final Outcome")
        explanation_parts.append(f"   {final_outcome}")

        return "\n".join(explanation_parts)

    def explain_performance_characteristics(
        self,
        generation_time: float,
        vram_usage: float,
        quality_estimate: Optional[float] = None,
        bottlenecks: Optional[list[str]] = None,
    ) -> str:
        """
        Explain performance characteristics of generation.

        Args:
            generation_time: Time taken in seconds
            vram_usage: VRAM used in GB
            quality_estimate: Estimated quality score (0-1)
            bottlenecks: Any performance bottlenecks

        Returns:
            Explanation text
        """
        explanation_parts = []

        explanation_parts.append("Performance Analysis:")
        explanation_parts.append(f"  • Generation time: {generation_time:.2f}s")
        explanation_parts.append(f"  • VRAM usage: {vram_usage:.2f}GB")

        if quality_estimate is not None:
            explanation_parts.append(f"  • Estimated quality: {quality_estimate:.2f}")

        if self.verbosity in (ExplanationVerbosity.DETAILED, ExplanationVerbosity.TECHNICAL):
            if bottlenecks:
                explanation_parts.append("\nBottlenecks:")
                for bottleneck in bottlenecks:
                    explanation_parts.append(f"  ⚠ {bottleneck}")

        return "\n".join(explanation_parts)

    def create_user_friendly_summary(
        self,
        result: GenerationResultProtocol,
        include_tips: bool = True,
    ) -> str:
        """
        Create a user-friendly summary of generation results.

        Args:
            result: GenerationResult
            include_tips: Whether to include tips for improvement

        Returns:
            User-friendly summary text
        """
        summary_parts = []

        summary_parts.append("✨ Generation Summary ✨\n")

        # What was generated
        summary_parts.append(f"Prompt: {result.metadata.prompt[:60]}...")
        summary_parts.append(
            f"Generated in {result.metadata.generation_time_seconds:.1f}s "
            f"using {result.metadata.peak_vram_gb:.1f}GB VRAM"
        )
        summary_parts.append("")

        # LoRAs used
        if result.metadata.loras_used:
            summary_parts.append("LoRAs Applied:")
            for lora in result.metadata.loras_used:
                summary_parts.append(f"  • {lora.name} (strength: {lora.alpha:.2f})")
        summary_parts.append("")

        # Parameters
        summary_parts.append("Parameters:")
        summary_parts.append(
            f"  • Quality: {result.metadata.steps} steps, "
            f"CFG {result.metadata.cfg_scale}"
        )
        summary_parts.append(
            f"  • Size: {result.metadata.width}×{result.metadata.height}"
        )
        summary_parts.append(f"  • Seed: {result.metadata.seed} (for reproducibility)")
        summary_parts.append("")

        # Tips
        if include_tips and self.verbosity != ExplanationVerbosity.MINIMAL:
            summary_parts.append("💡 Tips:")
            summary_parts.append(
                "  • To regenerate this exact image, use the same seed"
            )
            summary_parts.append(
                "  • To get variations, change the seed while keeping other params"
            )
            summary_parts.append(
                "  • Increase steps for higher quality (but slower generation)"
            )

        return "\n".join(summary_parts)

    def set_verbosity(self, verbosity: ExplanationVerbosity):
        """
        Change explanation verbosity level.

        Args:
            verbosity: New verbosity level
        """
        self.verbosity = verbosity
        logger.info(f"Explanation verbosity set to: {verbosity.value}")
