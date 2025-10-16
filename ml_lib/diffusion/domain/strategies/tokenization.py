"""Tokenization Strategies - Parse prompts respecting SD syntax.

Strategies for tokenizing prompts while respecting Stable Diffusion
emphasis syntax (parentheses, brackets, braces).
"""

import re
from typing import Optional

from ml_lib.diffusion.domain.interfaces.analysis_strategies import (
    ITokenizationStrategy,
)


class StableDiffusionTokenization(ITokenizationStrategy):
    """
    Tokenization strategy for Stable Diffusion prompts.

    Handles SD-specific syntax:
    - (emphasis) -> 1.1x weight
    - ((strong emphasis)) -> 1.21x weight
    - [de-emphasis] -> 0.9x weight
    - {attention} -> preserved
    - Commas as separators
    - AND keyword for blending
    """

    def tokenize(self, prompt: str) -> list[str]:
        """
        Tokenize prompt respecting SD syntax.

        Args:
            prompt: Prompt to tokenize

        Returns:
            List of tokens
        """
        # Remove extra whitespace
        prompt = " ".join(prompt.split())

        # Split by commas first (main separator in SD)
        parts = prompt.split(",")

        tokens = []
        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Handle AND keyword (blending)
            if " AND " in part.upper():
                # Split by AND and add as separate tokens
                and_parts = re.split(r"\s+AND\s+", part, flags=re.IGNORECASE)
                tokens.extend([p.strip() for p in and_parts if p.strip()])
            else:
                tokens.append(part)

        return tokens

    def extract_emphasis_map(self, prompt: str) -> dict[str, float]:
        """
        Extract emphasis weights from prompt.

        Handles:
        - (word) -> 1.1x
        - ((word)) -> 1.21x
        - (((word))) -> 1.331x
        - [word] -> 0.9x
        - [[word]] -> 0.81x

        Args:
            prompt: Prompt with SD emphasis syntax

        Returns:
            Dictionary mapping keywords to emphasis weights
        """
        emphasis_map: dict[str, float] = {}

        # Find all emphasized tokens (parentheses)
        # Pattern: one or more opening parens, content, same number closing parens
        emphasis_pattern = r"\(+([^()]+)\)+"
        for match in re.finditer(emphasis_pattern, prompt):
            content = match.group(1).strip()
            # Count depth (number of nested parentheses)
            depth = match.group(0).count("(")
            # Each level adds 0.1 (1.1x)
            weight = 1.0 + (depth * 0.1)
            emphasis_map[content] = weight

        # Find all de-emphasized tokens (brackets)
        deemphasis_pattern = r"\[+([^\[\]]+)\]+"
        for match in re.finditer(deemphasis_pattern, prompt):
            content = match.group(1).strip()
            # Count depth
            depth = match.group(0).count("[")
            # Each level subtracts 0.1 (0.9x)
            weight = 1.0 - (depth * 0.1)
            # Don't go below 0.1
            weight = max(weight, 0.1)
            emphasis_map[content] = weight

        return emphasis_map


class SimpleTokenization(ITokenizationStrategy):
    """
    Simple tokenization strategy that just splits on commas.

    Useful for testing or when SD syntax is not needed.
    """

    def tokenize(self, prompt: str) -> list[str]:
        """Tokenize by splitting on commas."""
        tokens = [t.strip() for t in prompt.split(",")]
        return [t for t in tokens if t]  # Filter empty

    def extract_emphasis_map(self, prompt: str) -> dict[str, float]:
        """No emphasis extraction for simple tokenization."""
        return {}


class AdvancedTokenization(ITokenizationStrategy):
    """
    Advanced tokenization with support for:
    - Weight modifiers: (word:1.5)
    - Negative emphasis: (word:-0.5)
    - Step scheduling: [word:0.5]
    - Blending: word1 AND word2
    - Alternating: [word1|word2]
    """

    def tokenize(self, prompt: str) -> list[str]:
        """Advanced tokenization with all SD features."""
        # Remove extra whitespace
        prompt = " ".join(prompt.split())

        # Split by commas first
        parts = prompt.split(",")

        tokens = []
        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Handle AND blending
            if " AND " in part.upper():
                and_parts = re.split(r"\s+AND\s+", part, flags=re.IGNORECASE)
                tokens.extend([p.strip() for p in and_parts if p.strip()])
            # Handle alternating [word1|word2]
            elif re.search(r"\[[^\[\]]+\|[^\[\]]+\]", part):
                # Extract alternating words
                alt_match = re.search(r"\[([^\[\]]+)\]", part)
                if alt_match:
                    alternates = alt_match.group(1).split("|")
                    tokens.extend([a.strip() for a in alternates if a.strip()])
            else:
                tokens.append(part)

        return tokens

    def extract_emphasis_map(self, prompt: str) -> dict[str, float]:
        """
        Extract emphasis with advanced features.

        Supports:
        - (word:1.5) -> explicit weight
        - (word) -> 1.1x
        - [word:0.5] -> step scheduling (treated as weight)
        """
        emphasis_map: dict[str, float] = {}

        # Explicit weight syntax: (word:1.5)
        weight_pattern = r"\(([^():]+):([0-9.]+)\)"
        for match in re.finditer(weight_pattern, prompt):
            content = match.group(1).strip()
            weight = float(match.group(2))
            emphasis_map[content] = weight

        # Standard parentheses emphasis
        emphasis_pattern = r"\(+([^()]+)\)+"
        for match in re.finditer(emphasis_pattern, prompt):
            content = match.group(1).strip()

            # Skip if already has explicit weight
            if ":" in content:
                continue

            depth = match.group(0).count("(")
            weight = 1.0 + (depth * 0.1)
            emphasis_map[content] = weight

        # Bracket de-emphasis
        deemphasis_pattern = r"\[+([^\[\]]+)\]+"
        for match in re.finditer(deemphasis_pattern, prompt):
            content = match.group(1).strip()

            # Skip alternating syntax [word1|word2]
            if "|" in content:
                continue

            # Check for step scheduling [word:0.5]
            if ":" in content:
                parts = content.split(":")
                if len(parts) == 2:
                    word = parts[0].strip()
                    try:
                        weight = float(parts[1])
                        emphasis_map[word] = weight
                    except ValueError:
                        pass
                continue

            depth = match.group(0).count("[")
            weight = 1.0 - (depth * 0.1)
            weight = max(weight, 0.1)
            emphasis_map[content] = weight

        return emphasis_map
