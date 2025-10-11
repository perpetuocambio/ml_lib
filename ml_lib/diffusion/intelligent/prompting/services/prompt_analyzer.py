"""Prompt analyzer using Ollama for semantic analysis."""

import json
import logging
import re
from typing import Optional
from pathlib import Path

from ml_lib.llm.entities.llm_prompt import LLMPrompt
from ml_lib.llm.providers.ollama_provider import OllamaProvider
from ml_lib.llm.entities.llm_provider_type import LLMProviderType
from ml_lib.llm.config.llm_provider_config import LLMProviderConfig

from ml_lib.diffusion.intelligent.prompting.entities import (
    PromptAnalysis,
    Intent,
    ArtisticStyle,
    ContentType,
    QualityLevel,
)
from ml_lib.diffusion.intelligent.prompting.handlers.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class PromptAnalyzer:
    """Analyzes prompts using Ollama for semantic understanding.

    Loads concept categories from external configuration.
    """

    def __init__(
        self,
        config: ConfigLoader | None = None,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "llama2",
        use_llm: bool = True,
    ):
        """
        Initialize prompt analyzer.

        Args:
            config: ConfigLoader with concept categories (if None, loads default)
            ollama_url: Ollama server URL
            model_name: Ollama model to use
            use_llm: Whether to use LLM for enhanced analysis
        """
        # Load configuration
        if config is None:
            from ml_lib.diffusion.intelligent.prompting.handlers.config_loader import get_default_config
            config = get_default_config()

        self.config = config
        self.CONCEPT_CATEGORIES = config.concept_categories
        self.use_llm = use_llm

        if use_llm:
            # Initialize Ollama provider
            config = LLMProviderConfig(
                provider_type=LLMProviderType.OLLAMA,
                model_name=model_name,
                api_endpoint=ollama_url,
                temperature=0.3,  # Lower temperature for more deterministic analysis
            )
            self.llm_provider = OllamaProvider(configuration=config)

            # Check availability
            if not self.llm_provider.is_available():
                logger.warning(
                    f"Ollama not available at {ollama_url}. "
                    "Falling back to rule-based analysis."
                )
                self.use_llm = False

        logger.info(
            f"PromptAnalyzer initialized (LLM: {self.use_llm}, model: {model_name})"
        )

    def analyze(self, prompt: str) -> PromptAnalysis:
        """
        Analyze a prompt comprehensively.

        Args:
            prompt: User prompt to analyze

        Returns:
            Complete prompt analysis
        """
        # 1. Tokenize
        tokens = self._tokenize(prompt)

        # 2. Extract concepts
        concepts = self._extract_concepts(tokens, prompt)

        # 3. Detect intent
        intent = self._detect_intent(concepts, tokens, prompt)

        # 4. Calculate complexity
        complexity = self._calculate_complexity(tokens, concepts)

        # 5. Build emphasis map
        emphasis = self._build_emphasis_map(tokens, prompt)

        return PromptAnalysis(
            original_prompt=prompt,
            tokens=tokens,
            detected_concepts=concepts,
            intent=intent,
            complexity_score=complexity,
            emphasis_map=emphasis,
        )

    def _tokenize(self, prompt: str) -> list[str]:
        """
        Tokenize prompt respecting SD syntax.

        Handles:
        - (emphasis)
        - [de-emphasis]
        - {attention}
        """
        # Remove special brackets but keep content
        cleaned = prompt
        cleaned = re.sub(r"[()\[\]{}]", " ", cleaned)

        # Split by comma and whitespace
        tokens = [t.strip() for t in cleaned.split(",") if t.strip()]

        # Also split individual words for detailed analysis
        words = []
        for token in tokens:
            words.extend(token.split())

        return tokens + words

    def _extract_concepts(
        self, tokens: list[str], full_prompt: str
    ) -> dict[str, list[str]]:
        """
        Extract concepts by category.

        If LLM is available, enhances extraction with semantic understanding.
        """
        concepts = {}

        # Rule-based extraction
        for category, keywords in self.CONCEPT_CATEGORIES.items():
            found = []
            for token in tokens:
                token_lower = token.lower()
                for keyword in keywords:
                    if keyword.lower() in token_lower:
                        found.append(token)
                        break

            if found:
                concepts[category] = list(set(found))

        # Enhance with LLM if available
        if self.use_llm and full_prompt:
            llm_concepts = self._llm_extract_concepts(full_prompt)
            # Merge LLM results
            for category, items in llm_concepts.items():
                if category in concepts:
                    concepts[category].extend(items)
                    concepts[category] = list(set(concepts[category]))
                else:
                    concepts[category] = items

        return concepts

    def _llm_extract_concepts(self, prompt: str) -> dict[str, list[str]]:
        """Use LLM to extract concepts semantically."""
        try:
            analysis_prompt = f"""Analyze this image generation prompt and extract key concepts:
"{prompt}"

Categorize the concepts into: character, style, content, setting, quality, lighting, camera, technical.
Respond ONLY with a JSON object like:
{{"character": ["concept1"], "style": ["concept2"], ...}}

JSON:"""

            llm_prompt = LLMPrompt(content=analysis_prompt, temperature=0.3)
            response = self.llm_provider.generate_response(llm_prompt)

            # Parse JSON from response
            content = response.content.strip()

            # Extract JSON if wrapped in code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            concepts = json.loads(content)
            return concepts

        except Exception as e:
            logger.warning(f"LLM concept extraction failed: {e}")
            return {}

    def _detect_intent(
        self, concepts: dict[str, list[str]], tokens: list[str], full_prompt: str
    ) -> Intent:
        """Detect artistic intent from concepts."""
        # Detect artistic style - ALWAYS default to photorealistic
        style_concepts = concepts.get("style", [])
        artistic_style = ArtisticStyle.PHOTOREALISTIC  # DEFAULT: Always photorealistic

        # Only check for explicit fantasy/sci-fi modifiers
        # Even fantasy must be photorealistic
        style_keywords = {
            "photorealistic": ArtisticStyle.PHOTOREALISTIC,
            "realistic": ArtisticStyle.PHOTOREALISTIC,
            "hyperrealistic": ArtisticStyle.PHOTOREALISTIC,
            "photo": ArtisticStyle.PHOTOREALISTIC,
        }

        for concept in style_concepts:
            concept_lower = concept.lower()
            for keyword, style in style_keywords.items():
                if keyword in concept_lower:
                    artistic_style = style
                    break

        # Override: If no style specified, assume photorealistic
        if not style_concepts:
            artistic_style = ArtisticStyle.PHOTOREALISTIC

        # Detect content type - Focus on people/couples
        content_type = ContentType.CHARACTER  # DEFAULT: Always character/portrait

        # Check for multiple subjects (couples/groups)
        if any(
            kw in full_prompt.lower()
            for kw in ["couple", "two women", "three women", "duo", "trio", "group"]
        ):
            content_type = ContentType.SCENE  # Multiple people = scene

        # Check for portrait focus
        elif any(
            kw in full_prompt.lower()
            for kw in ["portrait", "headshot", "face", "close-up"]
        ):
            content_type = ContentType.PORTRAIT

        # Default to character for single subject
        elif concepts.get("subjects"):
            content_type = ContentType.CHARACTER

        # Detect quality level
        quality_level = QualityLevel.MEDIUM

        quality_concepts = concepts.get("quality", [])
        if any(kw in " ".join(quality_concepts).lower() for kw in ["masterpiece"]):
            quality_level = QualityLevel.MASTERPIECE
        elif any(
            kw in " ".join(quality_concepts).lower() for kw in ["high", "detailed"]
        ):
            quality_level = QualityLevel.HIGH

        # Enhance with LLM
        if self.use_llm:
            llm_intent = self._llm_detect_intent(full_prompt)
            if llm_intent:
                return llm_intent

        confidence = 0.7 if artistic_style != ArtisticStyle.UNKNOWN else 0.5

        return Intent(
            artistic_style=artistic_style,
            content_type=content_type,
            quality_level=quality_level,
            confidence=confidence,
        )

    def _llm_detect_intent(self, prompt: str) -> Optional[Intent]:
        """Use LLM to detect intent."""
        try:
            intent_prompt = f"""Analyze this image generation prompt and determine the artistic intent:
"{prompt}"

Classify:
1. Artistic style: photorealistic, anime, cartoon, painting, sketch, abstract, concept_art, unknown
2. Content type: character, portrait, landscape, scene, object, abstract, unknown
3. Quality level: low, medium, high, masterpiece

Respond ONLY with JSON:
{{"artistic_style": "...", "content_type": "...", "quality_level": "..."}}

JSON:"""

            llm_prompt = LLMPrompt(content=intent_prompt, temperature=0.2)
            response = self.llm_provider.generate_response(llm_prompt)

            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            intent_data = json.loads(content)

            return Intent(
                artistic_style=ArtisticStyle(intent_data["artistic_style"]),
                content_type=ContentType(intent_data["content_type"]),
                quality_level=QualityLevel(intent_data["quality_level"]),
                confidence=0.9,  # High confidence from LLM
            )

        except Exception as e:
            logger.warning(f"LLM intent detection failed: {e}")
            return None

    def _calculate_complexity(
        self, tokens: list[str], concepts: dict[str, list[str]]
    ) -> float:
        """
        Calculate complexity score (0-1).

        Factors:
        - Number of tokens (25%)
        - Diversity of concepts (35%)
        - Specific keywords (40%)
        """
        # Lexical complexity
        lexical = min(len(tokens) / 100.0, 1.0)

        # Semantic complexity (concept diversity)
        semantic = min(len(concepts) / 10.0, 1.0)

        # Detail complexity
        detail_keywords = [
            "detailed",
            "intricate",
            "complex",
            "elaborate",
            "photorealistic",
            "masterpiece",
        ]
        detail_count = sum(
            1 for token in tokens if any(kw in token.lower() for kw in detail_keywords)
        )
        detail = min(detail_count / 5.0, 1.0)

        # Weighted score
        score = 0.25 * lexical + 0.35 * semantic + 0.40 * detail

        return min(max(score, 0.0), 1.0)

    def _build_emphasis_map(self, tokens: list[str], prompt: str) -> dict[str, float]:
        """
        Build emphasis map based on SD syntax.

        (word) = 1.1x emphasis
        ((word)) = 1.21x emphasis
        [word] = 0.9x de-emphasis
        """
        emphasis_map = {}

        # Count emphasis markers
        for token in tokens:
            base_token = re.sub(r"[()\[\]{}]", "", token).strip()
            if not base_token:
                continue

            # Count parentheses
            open_count = prompt.count(f"({token})")
            double_open = prompt.count(f"(({token}))")

            if double_open > 0:
                emphasis_map[base_token] = 1.21
            elif open_count > 0:
                emphasis_map[base_token] = 1.1
            elif f"[{token}]" in prompt:
                emphasis_map[base_token] = 0.9
            else:
                emphasis_map[base_token] = 1.0

        return emphasis_map
