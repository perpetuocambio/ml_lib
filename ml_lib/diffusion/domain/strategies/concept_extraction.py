"""Concept Extraction Strategies - Rule-based and LLM-enhanced.

Implementations of IConceptExtractionStrategy for extracting semantic
concepts from prompts using different approaches.
"""

from typing import Optional
import logging

from ml_lib.diffusion.domain.interfaces.analysis_strategies import (
    IConceptExtractionStrategy,
    ConceptExtractionResult,
)

logger = logging.getLogger(__name__)


class RuleBasedConceptExtraction(IConceptExtractionStrategy):
    """
    Rule-based concept extraction using keyword matching.

    Fast, reliable, no external dependencies.
    Good baseline for concept extraction.
    """

    def __init__(self, keyword_config: Optional[dict[str, list[str]]] = None):
        """
        Initialize with keyword configuration.

        Args:
            keyword_config: Dictionary mapping categories to keywords
                           e.g., {"character": ["girl", "boy"], ...}
        """
        self.keyword_config = keyword_config or self._get_default_keywords()

    def extract_concepts(
        self,
        prompt: str,
        tokens: list[str],
    ) -> ConceptExtractionResult:
        """Extract concepts using keyword matching."""
        prompt_lower = prompt.lower()
        concepts: dict[str, list[str]] = {}

        # Match keywords to categories
        for category, keywords in self.keyword_config.items():
            found_keywords = []
            for keyword in keywords:
                if keyword.lower() in prompt_lower:
                    found_keywords.append(keyword)

            if found_keywords:
                concepts[category] = found_keywords

        # Calculate confidence based on matches found
        total_categories = len(self.keyword_config)
        matched_categories = len(concepts)
        confidence = min(matched_categories / max(total_categories, 1), 1.0)

        return ConceptExtractionResult(
            concepts_by_category=concepts,
            confidence=confidence,
        )

    def _get_default_keywords(self) -> dict[str, list[str]]:
        """Get default keyword configuration."""
        return {
            "character": [
                "girl",
                "boy",
                "woman",
                "man",
                "person",
                "character",
                "anime girl",
                "anime boy",
            ],
            "style": [
                "photorealistic",
                "anime",
                "manga",
                "cartoon",
                "painting",
                "sketch",
                "abstract",
                "concept art",
                "digital art",
            ],
            "content": [
                "portrait",
                "landscape",
                "scene",
                "close-up",
                "wide shot",
                "full body",
                "headshot",
            ],
            "quality": [
                "masterpiece",
                "best quality",
                "high quality",
                "detailed",
                "ultra detailed",
                "8k",
                "4k",
                "hd",
            ],
        }


class LLMEnhancedConceptExtraction(IConceptExtractionStrategy):
    """
    LLM-enhanced concept extraction using semantic analysis.

    Uses Ollama/LLM for deeper understanding of prompts.
    Falls back to rule-based if LLM unavailable.
    """

    def __init__(
        self,
        llm_client,
        fallback_strategy: Optional[IConceptExtractionStrategy] = None,
    ):
        """
        Initialize with LLM client.

        Args:
            llm_client: Ollama client for LLM analysis
            fallback_strategy: Strategy to use if LLM fails
        """
        self.llm_client = llm_client
        self.fallback_strategy = fallback_strategy or RuleBasedConceptExtraction()

    def extract_concepts(
        self,
        prompt: str,
        tokens: list[str],
    ) -> ConceptExtractionResult:
        """Extract concepts using LLM semantic analysis."""
        try:
            # Try LLM extraction
            llm_prompt = self._build_llm_prompt(prompt)
            response = self.llm_client.generate(llm_prompt)

            # Parse LLM response
            concepts = self._parse_llm_response(response)

            return ConceptExtractionResult(
                concepts_by_category=concepts,
                confidence=0.9,  # High confidence for LLM
            )

        except Exception as e:
            logger.warning(f"LLM concept extraction failed: {e}. Using fallback.")
            # Fall back to rule-based
            return self.fallback_strategy.extract_concepts(prompt, tokens)

    def _build_llm_prompt(self, prompt: str) -> str:
        """Build prompt for LLM analysis."""
        return f"""Analyze this image generation prompt and extract key concepts:

Prompt: "{prompt}"

Categorize concepts into:
- character: People, animals, characters
- style: Art style, medium, aesthetic
- content: Scene type, composition, framing
- quality: Quality descriptors, detail level

Return in JSON format:
{{
    "character": ["keyword1", "keyword2"],
    "style": ["keyword1"],
    ...
}}"""

    def _parse_llm_response(self, response: str) -> dict[str, list[str]]:
        """Parse LLM JSON response."""
        import json

        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in LLM response")
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {}


class HybridConceptExtraction(IConceptExtractionStrategy):
    """
    Hybrid approach combining rule-based and LLM strategies.

    Uses rule-based for fast baseline, then enriches with LLM insights.
    Best of both worlds: speed + depth.
    """

    def __init__(
        self,
        rule_based_strategy: IConceptExtractionStrategy,
        llm_strategy: IConceptExtractionStrategy,
    ):
        """
        Initialize hybrid strategy.

        Args:
            rule_based_strategy: Fast rule-based extraction
            llm_strategy: Deep LLM extraction
        """
        self.rule_based = rule_based_strategy
        self.llm = llm_strategy

    def extract_concepts(
        self,
        prompt: str,
        tokens: list[str],
    ) -> ConceptExtractionResult:
        """Extract concepts using hybrid approach."""
        # Get rule-based results (fast baseline)
        rule_result = self.rule_based.extract_concepts(prompt, tokens)

        # Try to enhance with LLM (if available)
        try:
            llm_result = self.llm.extract_concepts(prompt, tokens)

            # Merge results (LLM enriches rule-based)
            merged_concepts = self._merge_concepts(
                rule_result.concepts_by_category,
                llm_result.concepts_by_category,
            )

            # Average confidence
            confidence = (rule_result.confidence + llm_result.confidence) / 2

            return ConceptExtractionResult(
                concepts_by_category=merged_concepts,
                confidence=confidence,
            )

        except Exception as e:
            logger.debug(f"LLM enhancement failed, using rule-based only: {e}")
            return rule_result

    def _merge_concepts(
        self,
        rule_concepts: dict[str, list[str]],
        llm_concepts: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """Merge rule-based and LLM concepts."""
        merged: dict[str, list[str]] = {}

        # Get all categories
        all_categories = set(rule_concepts.keys()) | set(llm_concepts.keys())

        for category in all_categories:
            rule_keywords = rule_concepts.get(category, [])
            llm_keywords = llm_concepts.get(category, [])

            # Combine and deduplicate
            combined = list(set(rule_keywords + llm_keywords))
            merged[category] = combined

        return merged
