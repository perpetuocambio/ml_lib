"""Intent Detection Strategies - Detect artistic intent from prompts.

Strategies for determining artistic style, content type, and quality level
from user prompts.
"""

import logging
from typing import Optional

from ml_lib.diffusion.domain.interfaces.analysis_strategies import (
    IIntentDetectionStrategy,
    IntentDetectionResult,
    ArtisticStyle,
    ContentType,
    QualityLevel,
)

logger = logging.getLogger(__name__)


class RuleBasedIntentDetection(IIntentDetectionStrategy):
    """
    Rule-based intent detection using keyword matching.

    Fast and reliable for common patterns.
    """

    def detect_intent(
        self,
        prompt: str,
        concepts: dict[str, list[str]],
        tokens: list[str],
    ) -> IntentDetectionResult:
        """Detect intent using rule-based keyword matching."""
        prompt_lower = prompt.lower()

        # Detect artistic style
        artistic_style = self._detect_artistic_style(prompt_lower)

        # Detect content type
        content_type = self._detect_content_type(prompt_lower, concepts)

        # Detect quality level
        quality_level = self._detect_quality_level(prompt_lower, concepts)

        # Calculate confidence based on matches
        confidence = self._calculate_confidence(prompt_lower, concepts)

        return IntentDetectionResult(
            artistic_style=artistic_style,
            content_type=content_type,
            quality_level=quality_level,
            confidence=confidence,
        )

    def _detect_artistic_style(self, prompt_lower: str) -> ArtisticStyle:
        """Detect artistic style from prompt."""
        style_keywords = {
            ArtisticStyle.PHOTOREALISTIC: [
                "photorealistic",
                "realistic",
                "photo",
                "photograph",
                "real life",
            ],
            ArtisticStyle.ANIME: ["anime", "manga", "animated"],
            ArtisticStyle.CARTOON: ["cartoon", "comic", "toon"],
            ArtisticStyle.PAINTING: [
                "painting",
                "painted",
                "oil painting",
                "watercolor",
                "acrylic",
            ],
            ArtisticStyle.SKETCH: ["sketch", "drawing", "pencil", "charcoal"],
            ArtisticStyle.ABSTRACT: ["abstract", "surreal", "avant-garde"],
            ArtisticStyle.CONCEPT_ART: ["concept art", "digital art", "illustration"],
        }

        # Check for explicit style keywords
        for style, keywords in style_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    return style

        # Default to photorealistic
        return ArtisticStyle.PHOTOREALISTIC

    def _detect_content_type(
        self, prompt_lower: str, concepts: dict[str, list[str]]
    ) -> ContentType:
        """Detect content type from prompt."""
        # Check for multi-subject indicators
        multi_subject_keywords = ["couple", "group", "duo", "multiple", "several"]
        if any(keyword in prompt_lower for keyword in multi_subject_keywords):
            return ContentType.SCENE

        # Check for portrait indicators
        portrait_keywords = [
            "portrait",
            "headshot",
            "face",
            "close-up",
            "closeup",
            "facial",
        ]
        if any(keyword in prompt_lower for keyword in portrait_keywords):
            return ContentType.PORTRAIT

        # Check for scene indicators
        scene_keywords = [
            "landscape",
            "scenery",
            "environment",
            "background",
            "setting",
            "location",
        ]
        if any(keyword in prompt_lower for keyword in scene_keywords):
            return ContentType.SCENE

        # Check for object focus
        object_keywords = ["object", "item", "product", "still life"]
        if any(keyword in prompt_lower for keyword in object_keywords):
            return ContentType.OBJECT

        # Check character concepts
        character_concepts = concepts.get("character", [])
        if character_concepts:
            return ContentType.CHARACTER

        # Default to scene
        return ContentType.SCENE

    def _detect_quality_level(
        self, prompt_lower: str, concepts: dict[str, list[str]]
    ) -> QualityLevel:
        """Detect quality level from prompt."""
        # Check for masterpiece indicators
        masterpiece_keywords = ["masterpiece", "best quality", "exceptional"]
        if any(keyword in prompt_lower for keyword in masterpiece_keywords):
            return QualityLevel.MASTERPIECE

        # Check for high quality indicators
        high_keywords = [
            "high quality",
            "detailed",
            "ultra detailed",
            "highly detailed",
            "8k",
            "4k",
            "uhd",
            "hd",
        ]
        if any(keyword in prompt_lower for keyword in high_keywords):
            return QualityLevel.HIGH

        # Check for low quality indicators
        low_keywords = ["low quality", "simple", "basic", "rough"]
        if any(keyword in prompt_lower for keyword in low_keywords):
            return QualityLevel.LOW

        # Check quality concepts
        quality_concepts = concepts.get("quality", [])
        if quality_concepts:
            # If has quality keywords, assume at least high
            return QualityLevel.HIGH

        # Default to medium
        return QualityLevel.MEDIUM

    def _calculate_confidence(
        self, prompt_lower: str, concepts: dict[str, list[str]]
    ) -> float:
        """Calculate confidence score based on matches."""
        confidence_score = 0.5  # Base confidence

        # Boost confidence if we found style keywords
        style_found = any(
            keyword in prompt_lower
            for keywords in [
                ["photorealistic", "anime", "painting", "sketch", "cartoon"]
            ]
            for keyword in keywords
        )
        if style_found:
            confidence_score += 0.2

        # Boost confidence if we found content keywords
        content_found = any(
            keyword in prompt_lower
            for keyword in ["portrait", "landscape", "character", "scene"]
        )
        if content_found:
            confidence_score += 0.2

        # Boost confidence if we have concept matches
        if len(concepts) > 0:
            confidence_score += 0.1

        return min(confidence_score, 1.0)


class LLMEnhancedIntentDetection(IIntentDetectionStrategy):
    """
    LLM-enhanced intent detection using semantic understanding.

    Uses LLM for deeper analysis, falls back to rule-based.
    """

    def __init__(
        self,
        llm_client,
        fallback_strategy: Optional[IIntentDetectionStrategy] = None,
    ):
        """
        Initialize with LLM client.

        Args:
            llm_client: Ollama client for LLM analysis
            fallback_strategy: Strategy to use if LLM fails
        """
        self.llm_client = llm_client
        self.fallback_strategy = fallback_strategy or RuleBasedIntentDetection()

    def detect_intent(
        self,
        prompt: str,
        concepts: dict[str, list[str]],
        tokens: list[str],
    ) -> IntentDetectionResult:
        """Detect intent using LLM semantic analysis."""
        try:
            # Try LLM detection
            llm_prompt = self._build_llm_prompt(prompt)
            response = self.llm_client.generate(llm_prompt)

            # Parse LLM response
            intent = self._parse_llm_response(response)

            return IntentDetectionResult(
                artistic_style=intent["style"],
                content_type=intent["content"],
                quality_level=intent["quality"],
                confidence=0.9,  # High confidence for LLM
            )

        except Exception as e:
            logger.warning(f"LLM intent detection failed: {e}. Using fallback.")
            # Fall back to rule-based
            return self.fallback_strategy.detect_intent(prompt, concepts, tokens)

    def _build_llm_prompt(self, prompt: str) -> str:
        """Build prompt for LLM analysis."""
        return f"""Analyze this image generation prompt and determine the artistic intent:

Prompt: "{prompt}"

Classify:
1. Artistic Style: photorealistic, anime, cartoon, painting, sketch, abstract, concept_art
2. Content Type: character, portrait, scene, object, abstract_concept
3. Quality Level: low, medium, high, masterpiece

Return in JSON format:
{{
    "style": "photorealistic",
    "content": "character",
    "quality": "high"
}}"""

    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM JSON response."""
        import json

        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                # Convert strings to enums
                return {
                    "style": self._parse_style(data.get("style", "photorealistic")),
                    "content": self._parse_content(data.get("content", "character")),
                    "quality": self._parse_quality(data.get("quality", "medium")),
                }
            else:
                logger.warning("No JSON found in LLM response")
                return self._get_defaults()
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return self._get_defaults()

    def _parse_style(self, style_str: str) -> ArtisticStyle:
        """Parse style string to enum."""
        style_map = {
            "photorealistic": ArtisticStyle.PHOTOREALISTIC,
            "anime": ArtisticStyle.ANIME,
            "cartoon": ArtisticStyle.CARTOON,
            "painting": ArtisticStyle.PAINTING,
            "sketch": ArtisticStyle.SKETCH,
            "abstract": ArtisticStyle.ABSTRACT,
            "concept_art": ArtisticStyle.CONCEPT_ART,
        }
        return style_map.get(style_str.lower(), ArtisticStyle.PHOTOREALISTIC)

    def _parse_content(self, content_str: str) -> ContentType:
        """Parse content string to enum."""
        content_map = {
            "character": ContentType.CHARACTER,
            "portrait": ContentType.PORTRAIT,
            "scene": ContentType.SCENE,
            "object": ContentType.OBJECT,
            "abstract_concept": ContentType.ABSTRACT_CONCEPT,
        }
        return content_map.get(content_str.lower(), ContentType.CHARACTER)

    def _parse_quality(self, quality_str: str) -> QualityLevel:
        """Parse quality string to enum."""
        quality_map = {
            "low": QualityLevel.LOW,
            "medium": QualityLevel.MEDIUM,
            "high": QualityLevel.HIGH,
            "masterpiece": QualityLevel.MASTERPIECE,
        }
        return quality_map.get(quality_str.lower(), QualityLevel.MEDIUM)

    def _get_defaults(self) -> dict:
        """Get default intent values."""
        return {
            "style": ArtisticStyle.PHOTOREALISTIC,
            "content": ContentType.CHARACTER,
            "quality": QualityLevel.MEDIUM,
        }
