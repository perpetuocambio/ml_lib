"""LoRA recommendation system based on prompt analysis."""

import logging
import numpy as np
from typing import Optional

from ml_lib.diffusion.intelligent.hub_integration.model_registry import ModelRegistry
from ml_lib.diffusion.intelligent.hub_integration.entities import (
    ModelMetadata,
    ModelType,
    BaseModel,
)
from ml_lib.diffusion.intelligent.prompting.entities import (
    PromptAnalysis,
    LoRARecommendation,
)
from ml_lib.diffusion.intelligent.prompting.handlers.config_loader import get_default_config

logger = logging.getLogger(__name__)


class LoRARecommender:
    """Recommends LoRAs based on prompt analysis.

    Optimized for PHOTOREALISTIC EXPLICIT adult content (30+).
    Filters out anime, cartoon, and stylized content.
    """

    def __init__(self, registry: ModelRegistry, config=None):
        """
        Initialize LoRA recommender.

        Args:
            registry: Model registry for accessing LoRA metadata
            config: PrompterConfig with configuration (if None, loads default)
        """
        self.registry = registry
        
        # Load configuration
        if config is None:
            config = get_default_config()
        self.config = config
        
        # Set up configurable values
        self.BLOCKED_TAGS = set(config.blocked_tags)
        self.PRIORITY_TAGS = set(config.priority_tags)
        self.ANATOMY_TAGS = set(config.anatomy_tags)
        self.PRIORITY_WEIGHT = config.scoring_weights.get("priority_score_weight", 0.25)
        self.ANATOMY_WEIGHT = config.scoring_weights.get("anatomy_score_weight", 0.20)
        self.KEYWORD_WEIGHT = config.scoring_weights.get("keyword_score_weight", 0.25)
        self.TAG_WEIGHT = config.scoring_weights.get("tag_score_weight", 0.20)
        self.POPULARITY_WEIGHT = config.scoring_weights.get("popularity_score_weight", 0.10)
        self.LORA_LIMITS = config.lora_limits
        
        logger.info("LoRARecommender initialized with configuration")

    def recommend(
        self,
        prompt_analysis: PromptAnalysis,
        base_model: str,
        max_loras: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> list[LoRARecommendation]:
        """
        Recommend LoRAs for the prompt.

        Algorithm:
        1. Filter LoRAs compatible with base_model
        2. Calculate relevance for each LoRA
        3. Score with multiple factors
        4. Resolve conflicts
        5. Balance weights
        6. Return top-K

        Args:
            prompt_analysis: Analyzed prompt
            base_model: Base model being used
            max_loras: Maximum LoRAs to recommend (uses config default if None)
            min_confidence: Minimum confidence threshold (uses config default if None)

        Returns:
            List of LoRA recommendations
        """
        # Use configuration defaults if not provided
        if max_loras is None:
            max_loras = self.LORA_LIMITS.get("max_loras", 3)
        if min_confidence is None:
            min_confidence = self.LORA_LIMITS.get("min_confidence", 0.5)
            
        # 1. Get compatible LoRAs
        base_model_enum = self._parse_base_model(base_model)
        candidates = self.registry.list_models(
            model_type=ModelType.LORA, base_model=base_model_enum, limit=200
        )

        if not candidates:
            logger.warning(f"No LoRAs found for base model: {base_model}")
            return []

        logger.info(f"Found {len(candidates)} candidate LoRAs before filtering")

        # 2. Filter out blocked content (anime, cartoon, underage)
        candidates = self._filter_blocked_loras(candidates)
        logger.info(f"After blocking filter: {len(candidates)} LoRAs remain")

        if not candidates:
            logger.warning("All LoRAs were filtered out (likely anime/cartoon)")
            return []

        # 3. Score each LoRA
        scored = []
        for lora in candidates:
            relevance = self._calculate_relevance(lora, prompt_analysis)

            if relevance >= min_confidence:
                scored.append((lora, relevance))

        # 3. Sort by relevance
        scored.sort(key=lambda x: x[1], reverse=True)

        # 4. Build recommendations
        recommendations = []
        for lora, score in scored[:max_loras * 2]:  # Get more than needed for filtering
            alpha = self._suggest_weight(lora, prompt_analysis, score)
            reasoning = self._generate_reasoning(lora, prompt_analysis, score)
            matching = self._find_matching_concepts(lora, prompt_analysis)

            rec = LoRARecommendation(
                lora_name=lora.name,
                lora_metadata=lora,
                confidence_score=score,
                suggested_alpha=alpha,
                matching_concepts=matching,
                reasoning=reasoning,
            )
            recommendations.append(rec)

        # 5. Resolve conflicts
        recommendations = self._resolve_conflicts(recommendations, max_loras)

        # 6. Rebalance weights
        recommendations = self._rebalance_weights(recommendations)

        return recommendations[:max_loras]

    def _filter_blocked_loras(self, loras: list[ModelMetadata]) -> list[ModelMetadata]:
        """
        Filter out LoRAs with blocked tags (anime, cartoon, underage).

        Args:
            loras: List of LoRA metadata

        Returns:
            Filtered list
        """
        filtered = []

        for lora in loras:
            # Check tags and description
            lora_text = f"{lora.name} {lora.description} {' '.join(lora.tags)}".lower()

            # Block if contains any blocked tag
            is_blocked = any(blocked in lora_text for blocked in self.BLOCKED_TAGS)

            if not is_blocked:
                filtered.append(lora)
            else:
                logger.debug(f"Blocked LoRA {lora.name}: contains anime/cartoon/underage tags")

        return filtered

    def _calculate_relevance(
        self, lora: ModelMetadata, analysis: PromptAnalysis
    ) -> float:
        """
        Calculate relevance score for a LoRA based on configuration.

        Uses configurable weights from configuration.
        """
        # Priority tag boost (photorealistic adult content)
        priority_score = self._priority_tag_score(lora)

        # Anatomy tag boost (detailed body parts)
        anatomy_score = self._anatomy_tag_score(lora, analysis)

        # Keyword matching in name/description
        keyword_score = self._keyword_match_score(lora, analysis)

        # Tag matching
        tag_score = self._tag_match_score(lora, analysis)

        # Popularity (normalized rating)
        popularity_score = min(lora.rating / 5.0, 1.0) if lora.rating > 0 else 0.5

        # Weighted combination using configurable weights
        relevance = (
            self.PRIORITY_WEIGHT * priority_score
            + self.ANATOMY_WEIGHT * anatomy_score
            + self.KEYWORD_WEIGHT * keyword_score
            + self.TAG_WEIGHT * tag_score
            + self.POPULARITY_WEIGHT * popularity_score
        )

        return min(max(relevance, 0.0), 1.0)

    def _priority_tag_score(self, lora: ModelMetadata) -> float:
        """
        Score based on priority tags (photorealistic, NSFW, mature).

        Returns:
            Score 0-1 based on priority tag presence
        """
        if not lora.tags:
            return 0.0

        lora_tags = set(tag.lower() for tag in lora.tags)
        lora_text = f"{lora.name} {lora.description}".lower()

        # Count priority tag matches
        matches = 0
        for priority_tag in self.PRIORITY_TAGS:
            if priority_tag in lora_text or any(priority_tag in tag for tag in lora_tags):
                matches += 1

        # Normalize (max ~5 priority tags expected)
        return min(matches / 5.0, 1.0)

    def _anatomy_tag_score(
        self, lora: ModelMetadata, analysis: PromptAnalysis
    ) -> float:
        """
        Score based on anatomical focus.

        Higher score if LoRA specializes in anatomical detail
        and prompt contains anatomical concepts.

        Returns:
            Score 0-1
        """
        # Check if prompt has anatomical focus
        anatomy_concepts = analysis.detected_concepts.get("anatomy", [])
        if not anatomy_concepts:
            return 0.0

        if not lora.tags:
            return 0.0

        lora_tags = set(tag.lower() for tag in lora.tags)
        lora_text = f"{lora.name} {lora.description}".lower()

        # Count anatomy tag matches
        matches = 0
        for anatomy_tag in self.ANATOMY_TAGS:
            if anatomy_tag in lora_text or any(anatomy_tag in tag for tag in lora_tags):
                matches += 1

        # Normalize (max ~3 anatomy tags expected)
        return min(matches / 3.0, 1.0)

    def _keyword_match_score(
        self, lora: ModelMetadata, analysis: PromptAnalysis
    ) -> float:
        """Score based on keyword matching."""
        lora_text = f"{lora.name} {lora.description}".lower()
        prompt_text = analysis.original_prompt.lower()

        # Extract significant words from prompt
        prompt_words = set(prompt_text.split())

        # Count matches
        matches = sum(1 for word in prompt_words if len(word) > 3 and word in lora_text)

        # Normalize
        return min(matches / max(len(prompt_words), 1), 1.0)

    def _tag_match_score(self, lora: ModelMetadata, analysis: PromptAnalysis) -> float:
        """Score based on tag matching."""
        if not lora.tags:
            return 0.0

        lora_tags = set(tag.lower() for tag in lora.tags)

        # Check concept matches
        matches = 0
        total_concepts = 0

        for category, concepts in analysis.detected_concepts.items():
            total_concepts += len(concepts)
            for concept in concepts:
                if any(concept.lower() in tag for tag in lora_tags):
                    matches += 1

        if total_concepts == 0:
            return 0.0

        return matches / total_concepts

    def _trigger_match_score(
        self, lora: ModelMetadata, analysis: PromptAnalysis
    ) -> float:
        """Score based on trigger word presence."""
        if not lora.trigger_words:
            return 0.5  # Neutral score if no triggers

        prompt_lower = analysis.original_prompt.lower()

        # Check if any trigger word is in prompt
        matches = sum(
            1 for trigger in lora.trigger_words if trigger.lower() in prompt_lower
        )

        return min(matches / len(lora.trigger_words), 1.0)

    def _suggest_weight(
        self, lora: ModelMetadata, analysis: PromptAnalysis, relevance: float
    ) -> float:
        """
        Suggest optimal weight (alpha) for the LoRA.

        Factors:
        - Recommended weight from metadata
        - Relevance score
        - Complexity of prompt
        """
        # Start with recommended weight or default from config
        base_weight = lora.recommended_weight if lora.recommended_weight else 0.7

        # Adjust by relevance
        weight = base_weight * (0.5 + 0.5 * relevance)

        # Adjust by complexity (reduce for complex prompts to avoid conflicts)
        if analysis.complexity_score > 0.7:
            weight *= 0.9

        # Use configurable limits
        min_weight = self.LORA_LIMITS.get("min_individual_weight", 0.3)
        max_weight = self.LORA_LIMITS.get("max_individual_weight", 1.2)

        # Clamp to reasonable range from config
        return min(max(weight, min_weight), max_weight)

    def _generate_reasoning(
        self, lora: ModelMetadata, analysis: PromptAnalysis, score: float
    ) -> str:
        """Generate human-readable reasoning for recommendation."""
        reasons = []

        # Matching concepts
        matching = self._find_matching_concepts(lora, analysis)
        if matching:
            reasons.append(f"Matches concepts: {', '.join(matching[:3])}")

        # Trigger words
        if lora.trigger_words:
            prompt_lower = analysis.original_prompt.lower()
            matched_triggers = [
                t for t in lora.trigger_words if t.lower() in prompt_lower
            ]
            if matched_triggers:
                reasons.append(f"Trigger words found: {', '.join(matched_triggers[:2])}")

        # Popularity
        if lora.rating >= 4.0:
            reasons.append(f"Highly rated ({lora.rating:.1f}/5.0)")

        # Relevance
        reasons.append(f"Relevance score: {score:.2f}")

        return " | ".join(reasons)

    def _find_matching_concepts(
        self, lora: ModelMetadata, analysis: PromptAnalysis
    ) -> list[str]:
        """Find concepts that match between LoRA and prompt."""
        matching = []

        lora_text = f"{lora.name} {lora.description} {' '.join(lora.tags)}".lower()

        for category, concepts in analysis.detected_concepts.items():
            for concept in concepts:
                if concept.lower() in lora_text:
                    matching.append(concept)

        return list(set(matching))

    def _resolve_conflicts(
        self, recommendations: list[LoRARecommendation], max_loras: int
    ) -> list[LoRARecommendation]:
        """
        Detect and resolve conflicts between LoRAs.

        Conflicts:
        - Style conflicts (anime vs photorealistic)
        - Overlapping concepts
        - Weight imbalance
        """
        resolved = []

        for i, rec in enumerate(recommendations):
            # Check compatibility with already selected LoRAs
            compatible = True

            for selected in resolved:
                if not rec.is_compatible_with(selected):
                    logger.debug(
                        f"Conflict detected: {rec.lora_name} vs {selected.lora_name}"
                    )
                    # Keep higher confidence
                    if rec.confidence_score > selected.confidence_score:
                        resolved.remove(selected)
                    else:
                        compatible = False
                        break

            if compatible:
                resolved.append(rec)

            # Stop if we have enough
            if len(resolved) >= max_loras:
                break

        return resolved

    def _rebalance_weights(
        self, recommendations: list[LoRARecommendation]
    ) -> list[LoRARecommendation]:
        """
        Rebalance LoRA weights to avoid over-influence.

        Total weight should not exceed configurable threshold to maintain base model character.
        """
        if not recommendations:
            return recommendations

        total_weight = sum(rec.suggested_alpha for rec in recommendations)

        # Use configurable max total weight
        max_total_weight = self.LORA_LIMITS.get("max_total_weight", 3.0)
        
        # If total exceeds threshold, scale down
        if total_weight > max_total_weight:
            scale_factor = max_total_weight / total_weight
            for rec in recommendations:
                rec.suggested_alpha *= scale_factor
                rec.suggested_alpha = round(rec.suggested_alpha, 2)

        return recommendations

    def _parse_base_model(self, base_model: str) -> BaseModel:
        """Parse base model string to enum."""
        base_model_lower = base_model.lower()

        if "sdxl" in base_model_lower or "xl" in base_model_lower:
            return BaseModel.SDXL
        elif "sd3" in base_model_lower:
            return BaseModel.SD3
        elif "pony" in base_model_lower:
            return BaseModel.PONY
        elif "sd2" in base_model_lower:
            return BaseModel.SD20
        elif "sd1" in base_model_lower or "1.5" in base_model_lower:
            return BaseModel.SD15
        else:
            return BaseModel.SDXL  # Default to SDXL
