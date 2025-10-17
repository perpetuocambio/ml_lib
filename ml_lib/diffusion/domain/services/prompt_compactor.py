"""Prompt compaction service for CLIP token limit handling."""

import logging
from typing import Optional, TYPE_CHECKING

from ml_lib.diffusion.models.content_tags import (
    TokenClassification,
    PromptCompactionResult,
    PromptTokenPriority,
    classify_token,
    analyze_nsfw_content,
)

if TYPE_CHECKING:
    from transformers import CLIPTokenizer

logger = logging.getLogger(__name__)


class PromptCompactor:
    """
    Compacts prompts to fit within CLIP's 77-token limit.

    Intelligently prioritizes content:
    1. CRITICAL: Core content (characters, main action)
    2. HIGH: NSFW acts, specific features
    3. MEDIUM: Context, setting, modifiers
    4. LOW: Quality tags
    5. DISCARD: Redundant tags

    Example:
        >>> compactor = PromptCompactor(max_tokens=77)
        >>> result = compactor.compact("very long prompt...")
        >>> print(result.compacted_prompt)
        >>> print(f"Removed {result.tokens_removed} tokens")
    """

    def __init__(
        self,
        max_tokens: int = 77,
        tokenizer_name: str = "openai/clip-vit-large-patch14",
    ):
        """
        Initialize compactor.

        Args:
            max_tokens: Maximum tokens allowed (CLIP default is 77)
            tokenizer_name: HuggingFace tokenizer to use for counting
        """
        self.max_tokens = max_tokens
        self.tokenizer_name = tokenizer_name
        self._tokenizer: Optional["CLIPTokenizer"] = None  # Lazy load

    def _get_tokenizer(self) -> "CLIPTokenizer":  # type: ignore
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            try:
                from transformers import CLIPTokenizer
                self._tokenizer = CLIPTokenizer.from_pretrained(self.tokenizer_name)
                logger.debug(f"Loaded tokenizer: {self.tokenizer_name}")
            except Exception as e:
                logger.warning(f"Failed to load CLIP tokenizer: {e}, using simple tokenizer")
                # Fallback to None - will use simple token counting
                self._tokenizer = None

        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using CLIP tokenizer.

        Args:
            text: Text to count tokens

        Returns:
            Number of tokens
        """
        tokenizer = self._get_tokenizer()

        if tokenizer:
            try:
                tokens = tokenizer.encode(text)
                return len(tokens)
            except Exception as e:
                logger.debug(f"Tokenizer encoding failed: {e}, using fallback")

        # Fallback: Estimate based on commas and words
        # Rough approximation: each comma-separated part ~= 2-3 tokens
        parts = text.split(",")
        return len(parts) * 2 + len(text.split()) // 2

    def compact(
        self,
        prompt: str,
        preserve_nsfw: bool = True,
        min_quality_tags: int = 2,
    ) -> PromptCompactionResult:
        """
        Compact prompt to fit within token limit.

        Args:
            prompt: Original prompt to compact
            preserve_nsfw: Whether to preserve NSFW content at high priority
            min_quality_tags: Minimum quality tags to keep (if space allows)

        Returns:
            PromptCompactionResult with compacted prompt and metadata
        """
        original_token_count = self.count_tokens(prompt)

        # If already under limit, return as-is
        if original_token_count <= self.max_tokens:
            logger.debug(f"Prompt already under limit ({original_token_count}/{self.max_tokens})")
            return PromptCompactionResult(
                original_prompt=prompt,
                original_token_count=original_token_count,
                max_tokens=self.max_tokens,
                compacted_prompt=prompt,
                compacted_token_count=original_token_count,
                kept_tokens=[],
                removed_tokens=[],
            )

        logger.info(
            f"Compacting prompt: {original_token_count} tokens -> {self.max_tokens} max"
        )

        # Analyze NSFW content first
        nsfw_analysis = analyze_nsfw_content(prompt)

        # Split prompt into parts (comma-separated)
        parts = [p.strip() for p in prompt.split(",") if p.strip()]

        # Classify each part
        classified_parts: list[TokenClassification] = []
        for part in parts:
            classification = classify_token(part)
            classified_parts.append(classification)

        # Sort by priority (highest first)
        classified_parts.sort(key=lambda x: x.priority.value, reverse=True)

        # Build compacted prompt incrementally
        kept_parts: list[TokenClassification] = []
        removed_parts: list[TokenClassification] = []
        current_prompt = ""
        quality_tags_kept = 0

        for classification in classified_parts:
            # Try adding this part
            test_prompt = current_prompt
            if test_prompt:
                test_prompt += ", " + classification.token
            else:
                test_prompt = classification.token

            test_token_count = self.count_tokens(test_prompt)

            # Decision logic
            should_add = False

            if test_token_count <= self.max_tokens:
                # Under limit - check if we should add based on priority
                if classification.priority.value >= PromptTokenPriority.MEDIUM.value:
                    # CRITICAL, HIGH, MEDIUM - always add
                    should_add = True
                elif classification.is_quality_tag:
                    # Quality tag - add if under min count
                    if quality_tags_kept < min_quality_tags:
                        should_add = True
                        quality_tags_kept += 1
                else:
                    # LOW priority - add if we have room
                    should_add = True
            else:
                # Over limit - only add CRITICAL items if we have none yet
                if not kept_parts and classification.priority == PromptTokenPriority.CRITICAL:
                    logger.warning("Adding CRITICAL token even though over limit")
                    should_add = True

            if should_add:
                current_prompt = test_prompt
                kept_parts.append(classification)
            else:
                removed_parts.append(classification)

        # Build result
        compacted_prompt = current_prompt
        compacted_token_count = self.count_tokens(compacted_prompt)

        result = PromptCompactionResult(
            original_prompt=prompt,
            original_token_count=original_token_count,
            max_tokens=self.max_tokens,
            compacted_prompt=compacted_prompt,
            compacted_token_count=compacted_token_count,
            kept_tokens=kept_parts,
            removed_tokens=removed_parts,
            nsfw_categories_found=nsfw_analysis.categories,
            quality_tags_removed=len([p for p in removed_parts if p.is_quality_tag]),
        )

        # Check if NSFW content was preserved
        if preserve_nsfw and nsfw_analysis.is_nsfw:
            kept_nsfw = [p for p in kept_parts if p.category is not None]
            if not kept_nsfw:
                result.nsfw_content_preserved = False
                result.add_warning(
                    "NSFW content detected but may have been removed during compaction!"
                )

        # Check if core content was preserved
        critical_parts = [
            p for p in classified_parts
            if p.priority == PromptTokenPriority.CRITICAL
        ]
        kept_critical = [
            p for p in kept_parts
            if p.priority == PromptTokenPriority.CRITICAL
        ]

        if critical_parts and not kept_critical:
            result.core_content_preserved = False
            result.add_warning("Core content may have been removed during compaction!")

        # Log result
        logger.info(
            f"Compacted: {original_token_count} -> {compacted_token_count} tokens "
            f"({result.compression_ratio:.1%} kept, {len(removed_parts)} parts removed)"
        )

        if result.warnings:
            for warning in result.warnings:
                logger.warning(warning)

        return result

    def compact_simple(self, prompt: str) -> str:
        """
        Simple compaction - just returns compacted prompt string.

        Args:
            prompt: Prompt to compact

        Returns:
            Compacted prompt string
        """
        result = self.compact(prompt)
        return result.compacted_prompt


# Convenience function
def compact_prompt(
    prompt: str,
    max_tokens: int = 77,
    preserve_nsfw: bool = True,
) -> PromptCompactionResult:
    """
    Convenience function to compact a prompt.

    Args:
        prompt: Prompt to compact
        max_tokens: Maximum tokens allowed
        preserve_nsfw: Whether to preserve NSFW content

    Returns:
        PromptCompactionResult
    """
    compactor = PromptCompactor(max_tokens=max_tokens)
    return compactor.compact(prompt, preserve_nsfw=preserve_nsfw)
