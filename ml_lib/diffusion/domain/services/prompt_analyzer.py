"""Prompt analyzer using Strategy Pattern for flexible analysis.

Refactored to use strategy pattern for:
- Concept extraction
- Intent detection
- Tokenization
- Prompt optimization
"""

import logging
from typing import Optional

from ml_lib.llm.entities.llm_prompt import LLMPrompt
from ml_lib.llm.providers.ollama_provider import OllamaProvider
from ml_lib.llm.entities.llm_provider_type import LLMProviderType
from ml_lib.llm.config.llm_provider_config import LLMProviderConfig

from ml_lib.diffusion.domain.value_objects_models import (
    PromptAnalysis,
    Intent,
)
from ml_lib.diffusion.domain.value_objects_models.value_objects import (
    ConceptMap,
    Concept,
    EmphasisMap,
    Emphasis,
)

# Import strategies
from ml_lib.diffusion.domain.interfaces.analysis_strategies import (
    IConceptExtractionStrategy,
    IIntentDetectionStrategy,
    ITokenizationStrategy,
    IOptimizationStrategy,
    QualityLevel,
)
from ml_lib.diffusion.domain.strategies import (
    # Concept extraction
    RuleBasedConceptExtraction,
    LLMEnhancedConceptExtraction,
    HybridConceptExtraction,
    # Intent detection
    RuleBasedIntentDetection,
    LLMEnhancedIntentDetection,
    # Tokenization
    StableDiffusionTokenization,
    AdvancedTokenization,
    # Optimization
    OptimizationStrategyFactory,
)

logger = logging.getLogger(__name__)


class PromptAnalyzer:
    """Analyzes prompts using Strategy Pattern for flexible analysis.

    Uses injected strategies for:
    - Concept extraction (rule-based, LLM-enhanced, hybrid)
    - Intent detection (rule-based, LLM-enhanced)
    - Tokenization (simple, SD syntax, advanced)
    - Optimization (model-specific)
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "llama2",
        use_llm: bool = True,
        # Strategy injection (optional, uses defaults if not provided)
        concept_strategy: Optional[IConceptExtractionStrategy] = None,
        intent_strategy: Optional[IIntentDetectionStrategy] = None,
        tokenization_strategy: Optional[ITokenizationStrategy] = None,
    ):
        """
        Initialize prompt analyzer with strategies.

        Args:
            ollama_url: Ollama server URL
            model_name: Ollama model to use
            use_llm: Whether to use LLM for enhanced analysis
            concept_strategy: Strategy for concept extraction (optional)
            intent_strategy: Strategy for intent detection (optional)
            tokenization_strategy: Strategy for tokenization (optional)
        """
        self.use_llm = use_llm
        self.llm_provider = None

        # Initialize LLM provider if needed
        if use_llm:
            try:
                config = LLMProviderConfig(
                    provider_type=LLMProviderType.OLLAMA,
                    model_name=model_name,
                    api_endpoint=ollama_url,
                    temperature=0.3,
                )
                self.llm_provider = OllamaProvider(configuration=config)

                # Check availability
                if not self.llm_provider.is_available():
                    logger.warning(
                        f"Ollama not available at {ollama_url}. "
                        "Falling back to rule-based analysis."
                    )
                    self.use_llm = False
                    self.llm_provider = None
            except Exception as e:
                logger.warning(f"Failed to initialize LLM provider: {e}")
                self.use_llm = False
                self.llm_provider = None

        # Initialize strategies (use defaults if not provided)
        self._init_strategies(concept_strategy, intent_strategy, tokenization_strategy)

        logger.info(
            f"PromptAnalyzer initialized (LLM: {self.use_llm}, model: {model_name})"
        )

    def _init_strategies(
        self,
        concept_strategy: Optional[IConceptExtractionStrategy],
        intent_strategy: Optional[IIntentDetectionStrategy],
        tokenization_strategy: Optional[ITokenizationStrategy],
    ):
        """Initialize strategies with smart defaults based on LLM availability."""

        # Tokenization: Use StableDiffusion tokenization by default
        if tokenization_strategy is not None:
            self.tokenization_strategy = tokenization_strategy
        else:
            self.tokenization_strategy = StableDiffusionTokenization()

        # Concept extraction: Use hybrid if LLM available, else rule-based
        if concept_strategy is not None:
            self.concept_strategy = concept_strategy
        else:
            if self.use_llm and self.llm_provider:
                # Hybrid combines rule-based + LLM
                rule_based = RuleBasedConceptExtraction()
                llm_enhanced = LLMEnhancedConceptExtraction(
                    llm_client=self.llm_provider
                )
                self.concept_strategy = HybridConceptExtraction(
                    rule_based_strategy=rule_based,
                    llm_strategy=llm_enhanced,
                )
            else:
                self.concept_strategy = RuleBasedConceptExtraction()

        # Intent detection: Use LLM-enhanced if available, else rule-based
        if intent_strategy is not None:
            self.intent_strategy = intent_strategy
        else:
            if self.use_llm and self.llm_provider:
                self.intent_strategy = LLMEnhancedIntentDetection(
                    llm_client=self.llm_provider
                )
            else:
                self.intent_strategy = RuleBasedIntentDetection()

    def analyze(self, prompt: str) -> PromptAnalysis:
        """
        Analyze a prompt comprehensively using strategies.

        Args:
            prompt: User prompt to analyze

        Returns:
            Complete prompt analysis
        """
        # 1. Tokenize using strategy
        tokens = self.tokenization_strategy.tokenize(prompt)

        # 2. Extract concepts using strategy
        concepts = self._extract_concepts_with_strategy(tokens, prompt)

        # 3. Detect intent using strategy
        intent = self._detect_intent_with_strategy(concepts, tokens, prompt)

        # 4. Calculate complexity
        complexity = self._calculate_complexity(tokens, concepts)

        # 5. Build emphasis map using tokenization strategy
        emphasis = self._build_emphasis_map_with_strategy(prompt)

        return PromptAnalysis(
            original_prompt=prompt,
            tokens=tokens,
            detected_concepts=concepts,
            intent=intent,
            complexity_score=complexity,
            emphasis_map=emphasis,
        )

    def _extract_concepts_with_strategy(
        self, tokens: list[str], full_prompt: str
    ) -> ConceptMap:
        """Extract concepts using injected strategy."""
        result = self.concept_strategy.extract_concepts(full_prompt, tokens)

        # Convert ConceptExtractionResult to ConceptMap
        # Concept expects 'name' (category) and '_keywords' (values)
        concepts_list = [
            Concept(name=category, _keywords=values)
            for category, values in result.concepts_by_category.items()
            if values  # Only include non-empty
        ]

        # ConceptMap validates non-empty, so handle empty case
        if not concepts_list:
            # Return a default concept to avoid validation error
            concepts_list = [Concept(name="unknown", _keywords=["none"])]

        return ConceptMap(_concepts=concepts_list)

    def _detect_intent_with_strategy(
        self, concepts: ConceptMap, tokens: list[str], full_prompt: str
    ) -> Intent:
        """Detect intent using injected strategy."""
        # Convert ConceptMap to dict for strategy interface
        concepts_dict = {}
        for concept in concepts:
            # Use 'name' (category) and get_keywords() for values
            concepts_dict[concept.name] = concept.get_keywords()

        # Use strategy to detect intent
        result = self.intent_strategy.detect_intent(
            prompt=full_prompt, concepts=concepts_dict, tokens=tokens
        )

        # Convert strategy result to Intent model
        return Intent(
            artistic_style=result.artistic_style,
            content_type=result.content_type,
            quality_level=result.quality_level,
            confidence=result.confidence,
        )

    def _build_emphasis_map_with_strategy(self, prompt: str) -> EmphasisMap:
        """Build emphasis map using tokenization strategy."""
        emphasis_dict = self.tokenization_strategy.extract_emphasis_map(prompt)

        # Convert dict to EmphasisMap
        emphases = [
            Emphasis(keyword=keyword, weight=weight)
            for keyword, weight in emphasis_dict.items()
        ]

        # EmphasisMap validates non-empty, handle empty case
        if not emphases:
            # Return default neutral emphasis
            emphases = [Emphasis(keyword="neutral", weight=1.0)]

        return EmphasisMap(_emphases=emphases)

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
        semantic = min(concepts.concept_count / 10.0, 1.0)

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

    def optimize_for_model(
        self,
        prompt: str,
        negative_prompt: str,
        base_model_architecture: str,
        quality: str = "balanced",
    ) -> tuple[str, str]:
        """
        Optimize prompts for specific model architecture using strategies.

        Adds quality tags, normalizes weights, and formats according to model requirements.

        Note: This method does NOT perform prompt compaction. If you need to compact
        the prompt to CLIP's 77-token limit, use PromptCompactor directly after calling
        this method.

        Args:
            prompt: User's positive prompt
            negative_prompt: User's negative prompt
            base_model_architecture: Model type (SDXL, Pony V6, SD 1.5, etc.)
            quality: Quality level (fast, balanced, high, ultra)

        Returns:
            Tuple of (optimized_positive, optimized_negative)
        """
        # Map quality string to QualityLevel enum
        quality_map = {
            "fast": QualityLevel.LOW,
            "balanced": QualityLevel.MEDIUM,
            "high": QualityLevel.HIGH,
            "ultra": QualityLevel.MASTERPIECE,
        }
        quality_level = quality_map.get(quality, QualityLevel.MEDIUM)

        # Get strategy for this architecture
        try:
            strategy = OptimizationStrategyFactory.create(base_model_architecture)
        except ValueError as e:
            logger.warning(
                f"Unknown architecture '{base_model_architecture}': {e}. "
                "Defaulting to SDXL."
            )
            strategy = OptimizationStrategyFactory.create("SDXL")

        # Use strategy to optimize
        result = strategy.optimize(
            prompt=prompt,
            negative_prompt=negative_prompt,
            quality_level=quality_level,
        )

        return result.optimized_prompt, result.optimized_negative
