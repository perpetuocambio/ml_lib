"""Prompt analyzer using Ollama for semantic analysis."""

import json
import logging
import re
from typing import Optional, Protocol
from pathlib import Path

from ml_lib.llm.entities.llm_prompt import LLMPrompt
from ml_lib.llm.providers.ollama_provider import OllamaProvider
from ml_lib.llm.entities.llm_provider_type import LLMProviderType
from ml_lib.llm.config.llm_provider_config import LLMProviderConfig

from ml_lib.diffusion.models import (
    PromptAnalysis,
    Intent,
    ArtisticStyle,
    ContentType,
    QualityLevel,
)
from ml_lib.diffusion.models.value_objects import (
    ConceptMap,
    Concept,
    EmphasisMap,
    Emphasis,
)

logger = logging.getLogger(__name__)


class ConfigProtocol(Protocol):
    """Protocol for configuration dict."""
    def get(self, key: str, default=None) -> list[str]:
        """Get configuration value."""
        ...
    def items(self) -> list[tuple[str, list[str]]]:
        """Get all items."""
        ...


class PromptAnalyzer:
    """Analyzes prompts using Ollama for semantic understanding.

    Loads concept categories from external configuration.
    """

    def __init__(
        self,
        config: Optional[ConfigProtocol] = None,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "llama2",
        use_llm: bool = True,
    ):
        """
        Initialize prompt analyzer.

        Args:
            config: Optional concept categories config (if None, uses defaults)
            ollama_url: Ollama server URL
            model_name: Ollama model to use
            use_llm: Whether to use LLM for enhanced analysis
        """
        # Define default concept categories as explicit attributes
        self._character_keywords = ["woman", "man", "person", "character"]
        self._style_keywords = ["photorealistic", "anime", "cartoon", "realistic"]
        self._content_keywords = ["portrait", "scene", "landscape"]
        self._quality_keywords = ["masterpiece", "high quality", "detailed"]

        # Store config if provided
        self.config = config
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
    ) -> ConceptMap:
        """
        Extract concepts by category.

        If LLM is available, enhances extraction with semantic understanding.
        """
        # Build category-keyword mapping
        categories = {
            "character": self._character_keywords,
            "style": self._style_keywords,
            "content": self._content_keywords,
            "quality": self._quality_keywords,
        }

        concepts_list: list[Concept] = []

        # Rule-based extraction
        for category, keywords in categories.items():
            found = []
            for token in tokens:
                token_lower = token.lower()
                for keyword in keywords:
                    if keyword.lower() in token_lower:
                        found.append(token)
                        break

            if found:
                # Remove duplicates
                unique_values = list(set(found))
                concepts_list.append(Concept(category=category, values=unique_values))

        # Enhance with LLM if available
        if self.use_llm and full_prompt:
            llm_concept_map = self._llm_extract_concepts(full_prompt)
            # Merge LLM results
            for llm_concept in llm_concept_map.concepts:
                # Find existing concept with same category
                existing = None
                for i, c in enumerate(concepts_list):
                    if c.category == llm_concept.category:
                        existing = i
                        break

                if existing is not None:
                    # Merge values
                    merged_values = list(set(concepts_list[existing].values + llm_concept.values))
                    concepts_list[existing] = Concept(
                        category=llm_concept.category,
                        values=merged_values
                    )
                else:
                    # Add new category
                    concepts_list.append(llm_concept)

        return ConceptMap(concepts=concepts_list)

    def _llm_extract_concepts(self, prompt: str) -> ConceptMap:
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

            concepts_dict = json.loads(content)

            # Convert dict to ConceptMap
            concepts_list = [
                Concept(category=category, values=values)
                for category, values in concepts_dict.items()
                if values  # Only include non-empty categories
            ]

            return ConceptMap(concepts=concepts_list)

        except Exception as e:
            logger.warning(f"LLM concept extraction failed: {e}")
            return ConceptMap(concepts=[])

    def _detect_intent(
        self, concepts: ConceptMap, tokens: list[str], full_prompt: str
    ) -> Intent:
        """Detect artistic intent from concepts."""
        # Detect artistic style - ALWAYS default to photorealistic
        style_concepts = concepts.get_category("style")
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
        elif concepts.get_category("subjects"):
            content_type = ContentType.CHARACTER

        # Detect quality level
        quality_level = QualityLevel.MEDIUM

        quality_concepts = concepts.get_category("quality")
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
        self, tokens: list[str], concepts: ConceptMap
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
        semantic = min(len(concepts.concepts) / 10.0, 1.0)

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

    def _build_emphasis_map(self, tokens: list[str], prompt: str) -> EmphasisMap:
        """
        Build emphasis map based on SD syntax.

        (word) = 1.1x emphasis
        ((word)) = 1.21x emphasis
        [word] = 0.9x de-emphasis
        """
        emphases: list[Emphasis] = []

        # Count emphasis markers
        for token in tokens:
            base_token = re.sub(r"[()\[\]{}]", "", token).strip()
            if not base_token:
                continue

            # Count parentheses
            open_count = prompt.count(f"({token})")
            double_open = prompt.count(f"(({token}))")

            weight = 1.0
            if double_open > 0:
                weight = 1.21
            elif open_count > 0:
                weight = 1.1
            elif f"[{token}]" in prompt:
                weight = 0.9

            emphases.append(Emphasis(keyword=base_token, weight=weight))

        return EmphasisMap(emphases=emphases)
