"""Value objects for processed prompts with user feedback."""

from dataclasses import dataclass, field

from ml_lib.diffusion.models.content_tags import (
    TokenClassification,
    PromptTokenPriority,
)


@dataclass
class ProcessedPrompt:
    """
    Result of prompt processing with user feedback information.

    This value object encapsulates the complete transformation of a user's
    original prompt into a final processed version, including all metadata
    about what was changed and why.
    """

    # Original input
    original: str
    original_token_count: int

    # Final processed output
    final: str
    final_token_count: int

    # Modification tracking
    was_modified: bool
    modifications: list[str] = field(default_factory=list)
    removed_tokens: list[TokenClassification] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Metadata
    architecture: str = ""  # Target model architecture (SDXL, Pony, etc.)
    quality_level: str = ""  # Quality preset used (fast, balanced, high, ultra)

    @property
    def has_critical_loss(self) -> bool:
        """
        Check if critical content was lost during processing.

        Returns:
            True if any CRITICAL priority tokens were removed
        """
        return any(
            t.priority == PromptTokenPriority.CRITICAL
            for t in self.removed_tokens
        )

    @property
    def has_high_priority_loss(self) -> bool:
        """
        Check if high-priority content was lost during processing.

        Returns:
            True if any HIGH priority tokens were removed
        """
        return any(
            t.priority == PromptTokenPriority.HIGH
            for t in self.removed_tokens
        )

    @property
    def token_reduction_ratio(self) -> float:
        """
        Calculate compression ratio.

        Returns:
            Ratio of final to original token count (0-1)
        """
        if self.original_token_count == 0:
            return 1.0
        return self.final_token_count / self.original_token_count

    @property
    def tokens_removed(self) -> int:
        """
        Number of tokens removed during processing.

        Returns:
            Count of removed tokens
        """
        return self.original_token_count - self.final_token_count

    def add_modification(self, modification: str) -> None:
        """
        Add a modification description.

        Args:
            modification: Human-readable description of what was changed
        """
        self.modifications.append(modification)

    def add_warning(self, warning: str) -> None:
        """
        Add a warning message.

        Args:
            warning: Warning message to show to user
        """
        self.warnings.append(warning)

    def get_user_message(self, verbose: bool = False) -> str:
        """
        Generate user-friendly feedback message.

        Args:
            verbose: If True, include detailed information about removed tokens

        Returns:
            Formatted message describing prompt processing
        """
        if not self.was_modified:
            return "✓ Prompt processed without changes"

        parts = []
        parts.append("⚠ Prompt was modified during processing:")
        parts.append(f"  • Tokens: {self.original_token_count} → {self.final_token_count}")

        # Show compression ratio if significant
        if self.token_reduction_ratio < 0.8:
            percentage = int((1 - self.token_reduction_ratio) * 100)
            parts.append(f"  • Reduction: {percentage}% compressed")

        # Show removed content
        if self.removed_tokens:
            removed_text = ", ".join(t.token for t in self.removed_tokens[:5])
            parts.append(f"  • Removed: {removed_text}")
            if len(self.removed_tokens) > 5:
                parts.append(f"    ... and {len(self.removed_tokens) - 5} more")

        # Critical warnings
        if self.has_critical_loss:
            parts.append("  ⚠ WARNING: Critical content was removed!")
            parts.append("    The generated image may not match your expectations.")

        if self.has_high_priority_loss and not self.has_critical_loss:
            parts.append("  ⚠ Note: Some high-priority content was removed.")

        # Show specific warnings
        if self.warnings:
            parts.append("  • Warnings:")
            for warning in self.warnings:
                parts.append(f"    - {warning}")

        # Verbose mode: show modifications
        if verbose and self.modifications:
            parts.append("  • Modifications:")
            for mod in self.modifications:
                parts.append(f"    - {mod}")

        return "\n".join(parts)

    def get_short_summary(self) -> str:
        """
        Get brief one-line summary.

        Returns:
            Short summary of processing result
        """
        if not self.was_modified:
            return "No changes"

        if self.has_critical_loss:
            return f"⚠ CRITICAL: {self.tokens_removed} tokens removed"

        if self.has_high_priority_loss:
            return f"⚠ {self.tokens_removed} tokens removed (includes high-priority content)"

        return f"{self.tokens_removed} tokens removed"

    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "original": self.original,
            "final": self.final,
            "original_token_count": self.original_token_count,
            "final_token_count": self.final_token_count,
            "was_modified": self.was_modified,
            "modifications": self.modifications,
            "removed_tokens": [
                {
                    "token": t.token,
                    "priority": t.priority.name,
                    "category": t.category.value if t.category else None,
                }
                for t in self.removed_tokens
            ],
            "warnings": self.warnings,
            "architecture": self.architecture,
            "quality_level": self.quality_level,
            "has_critical_loss": self.has_critical_loss,
            "has_high_priority_loss": self.has_high_priority_loss,
            "token_reduction_ratio": self.token_reduction_ratio,
        }
