"""Parameter optimization for image generation."""

import logging
import numpy as np
from dataclasses import dataclass

from ml_lib.diffusion.models import (
    PromptAnalysis,
    OptimizedParameters,
    Priority,
    ArtisticStyle,
)
from ml_lib.diffusion.models.value_objects import Resolution

logger = logging.getLogger(__name__)


@dataclass
class GenerationConstraints:
    """Constraints for generation."""

    max_time_seconds: float = 120.0
    max_vram_gb: float = 16.0
    priority: Priority = Priority.BALANCED


class ParameterOptimizer:
    """Optimizes generation parameters based on prompt analysis."""

    def __init__(self, config=None):
        """
        Initialize parameter optimizer.

        Args:
            config: Optional config object (if None, uses defaults)
        """
        # Store config or use defaults
        self.config = config

        # Define defaults - using explicit values instead of dicts
        self._min_steps = 20
        self._max_steps = 80
        self._min_cfg = 7.0
        self._max_cfg = 15.0
        self._min_resolution = Resolution(width=768, height=768)
        self._max_resolution = Resolution(width=1536, height=1536)

        # VRAM presets
        self._low_vram_max = Resolution(width=768, height=768)
        self._medium_vram_max = Resolution(width=1024, height=1024)
        self._high_vram_max = Resolution(width=1216, height=832)

        # Detail presets
        self._base_steps = 35
        self._base_cfg = 9.0
        self._base_resolution = Resolution(width=1024, height=1024)

        # Group profiles
        self._trio_resolution = Resolution(width=1280, height=896)
        self._couple_resolution = Resolution(width=1024, height=1024)

        # Sampler defaults
        self._sdxl_sampler = "DPM++ 2M Karras"
        self._sd20_sampler = "DPM++ 2M"
        self._default_clip_skip = 1

    def optimize(
        self,
        prompt_analysis: PromptAnalysis,
        constraints: GenerationConstraints | None = None,
    ) -> OptimizedParameters:
        """
        Optimize parameters based on prompt analysis and constraints.

        Multi-objective optimization:
        - Maximize quality
        - Minimize time
        - Respect VRAM constraints
        - Maximize prompt fidelity

        Args:
            prompt_analysis: Analyzed prompt
            constraints: Generation constraints

        Returns:
            Optimized parameters
        """
        constraints = constraints or GenerationConstraints()

        # Calculate each parameter
        steps = self._optimize_steps(prompt_analysis, constraints)
        cfg = self._optimize_cfg(prompt_analysis)
        resolution = self._optimize_resolution(prompt_analysis, constraints)
        sampler = self._select_sampler(prompt_analysis, constraints)
        clip_skip = self._determine_clip_skip(prompt_analysis)

        # Estimate resources
        vram = self._estimate_vram(resolution, steps)
        time = self._estimate_time(resolution, steps, sampler)
        quality = self._estimate_quality(steps, cfg)

        return OptimizedParameters(
            num_steps=steps,
            guidance_scale=cfg,
            width=resolution.width,
            height=resolution.height,
            sampler_name=sampler,
            clip_skip=clip_skip,
            estimated_time_seconds=time,
            estimated_vram_gb=vram,
            estimated_quality_score=quality,
            optimization_strategy=self._get_strategy_name(constraints),
            confidence=0.85,
        )

    def _optimize_steps(
        self, analysis: PromptAnalysis, constraints: GenerationConstraints
    ) -> int:
        """
        Optimize number of inference steps from configuration.

        Explicit anatomical detail requires MORE steps.
        """
        # Use base from defaults
        base = self._base_steps

        # Complexity adjustment from configuration
        complexity_bonus = 0
        if analysis.complexity_score < 0.3:
            complexity_bonus = 5
        elif analysis.complexity_score < 0.5:
            complexity_bonus = 10
        elif analysis.complexity_score < 0.7:
            complexity_bonus = 15
        elif analysis.complexity_score < 0.9:
            complexity_bonus = 25
        else:
            complexity_bonus = 35  # Maximum for very complex anatomy

        # Priority adjustment
        quality_bonus = 0
        speed_penalty = 0

        if constraints.priority == Priority.SPEED:
            speed_penalty = 10
        elif constraints.priority == Priority.QUALITY:
            quality_bonus = 20  # INCREASED for explicit quality

        # Quality level adjustment
        if analysis.intent and analysis.intent.quality_level.value == "masterpiece":
            quality_bonus += 15  # INCREASED

        # NEW: Anatomy bonus (explicit anatomical detail needs more steps)
        anatomy_bonus = 0
        anatomy_concepts = analysis.detected_concepts.get("anatomy", [])
        anatomy_count = len(anatomy_concepts)

        if anatomy_count > 8:
            anatomy_bonus = 20  # Extremely detailed anatomy
        elif anatomy_count > 5:
            anatomy_bonus = 15
        elif anatomy_count > 3:
            anatomy_bonus = 10

        # NEW: Activity bonus (sexual acts need more steps for accuracy)
        activity_concepts = analysis.detected_concepts.get("activity", [])
        activity_text = " ".join(activity_concepts).lower()
        activity_bonus = 0

        if any(kw in activity_text for kw in ["sex", "intercourse", "penetration"]):
            activity_bonus = 15
        elif any(kw in activity_text for kw in ["intimate", "erotic"]):
            activity_bonus = 10

        steps = base + complexity_bonus + quality_bonus + anatomy_bonus + activity_bonus - speed_penalty

        # Use configured range
        return int(np.clip(steps, self._min_steps, self._max_steps))

    def _optimize_cfg(self, analysis: PromptAnalysis) -> float:
        """
        Optimize CFG (guidance) scale from configuration.

        Uses configurable ranges and base values.
        """
        # START with configured base
        base_cfg = self._base_cfg

        # Adjust for complexity (more details = higher CFG)
        if analysis.complexity_score > 0.8:
            base_cfg += 2.0  # Very complex
        elif analysis.complexity_score > 0.7:
            base_cfg += 1.5
        elif analysis.complexity_score > 0.5:
            base_cfg += 1.0

        # Adjust for quality keywords
        quality_concepts = analysis.detected_concepts.get("quality", [])
        if any("detailed" in q.lower() or "hyperrealistic" in q.lower() or "8k" in q.lower()
               for q in quality_concepts):
            base_cfg += 1.0

        # CRITICAL: Anatomy focus (needs precise control)
        anatomy_concepts = analysis.detected_concepts.get("anatomy", [])
        anatomy_count = len(anatomy_concepts)

        if anatomy_count > 8:
            base_cfg += 2.5  # Extremely detailed anatomy
        elif anatomy_count > 5:
            base_cfg += 2.0  # Very detailed anatomy
        elif anatomy_count > 3:
            base_cfg += 1.5  # Detailed anatomy

        # EXPLICIT: Sexual activity needs higher CFG for positioning accuracy
        activity_concepts = analysis.detected_concepts.get("activity", [])
        activity_text = " ".join(activity_concepts).lower()

        if any(kw in activity_text for kw in ["sex", "intercourse", "penetration", "oral"]):
            base_cfg += 2.0  # Sexual acts
        elif any(kw in activity_text for kw in ["intimate", "erotic", "sensual", "touching"]):
            base_cfg += 1.5  # Intimate acts

        # EXPLICIT: Nudity and exposure benefit from higher CFG
        clothing_concepts = analysis.detected_concepts.get("clothing", [])
        if any(kw in " ".join(clothing_concepts).lower()
               for kw in ["nude", "naked", "exposed", "spread"]):
            base_cfg += 1.0

        # Age indicators (mature = more CFG for accurate age rendering)
        age_attrs = analysis.detected_concepts.get("age_attributes", [])
        if age_attrs:
            base_cfg += 0.5

        # Physical details (skin, pores, etc.) need higher CFG
        physical_details = analysis.detected_concepts.get("physical_details", [])
        if len(physical_details) > 3:
            base_cfg += 1.0

        # Use configured range
        return round(np.clip(base_cfg, self._min_cfg, self._max_cfg), 1)

    def _optimize_resolution(
        self, analysis: PromptAnalysis, constraints: GenerationConstraints
    ) -> Resolution:
        """
        Optimize resolution from configuration.

        Considerations:
        - Content type (portrait vs scene)
        - Group size (single, couple, trio)
        - Anatomical detail focus
        - VRAM constraints
        """
        # Use configured base resolution
        base_width = self._base_resolution.width
        base_height = self._base_resolution.height

        # Adjust for content type and group size using configuration
        if analysis.intent:
            content = analysis.intent.content_type.value

            # Check for group scenes
            subjects = analysis.detected_concepts.get("subjects", [])
            subject_text = " ".join(subjects).lower()

            # Detect group size and use configuration
            is_trio = any(kw in subject_text for kw in ["three", "trio", "group"])
            is_couple = any(kw in subject_text for kw in ["couple", "two", "duo"])

            if is_trio:
                # Trio - use configured resolution
                base_width = self._trio_resolution.width
                base_height = self._trio_resolution.height
            elif is_couple:
                # Couple - use configured resolution
                base_width = self._couple_resolution.width
                base_height = self._couple_resolution.height
                # Adjust for activity
                activity_concepts = analysis.detected_concepts.get("activity", [])
                if any("lying" in a or "horizontal" in a for a in activity_concepts):
                    base_width = 1216
                    base_height = 832
            elif content in ["portrait", "character"]:
                # Single portrait
                base_width = 896
                base_height = 1152
            elif content == "scene":
                # Scene
                base_width = 1152
                base_height = 896

        # NEW: Increase resolution for high anatomical detail
        anatomy_count = len(analysis.detected_concepts.get("anatomy", []))
        physical_details = len(analysis.detected_concepts.get("physical_details", []))

        if anatomy_count > 8 or physical_details > 5:
            # Extreme detail = increase by 12.5%
            base_width = int(base_width * 1.125)
            base_height = int(base_height * 1.125)
            # Round to nearest 64
            base_width = (base_width // 64) * 64
            base_height = (base_height // 64) * 64
        elif anatomy_count > 5:
            # High detail = increase by ~6%
            base_width = int(base_width * 1.0625)
            base_height = int(base_height * 1.0625)
            base_width = (base_width // 64) * 64
            base_height = (base_height // 64) * 64

        # Adjust for VRAM constraints using configured presets
        if constraints.max_vram_gb < 8:
            # Use low VRAM preset
            base_width = min(base_width, self._low_vram_max.width)
            base_height = min(base_height, self._low_vram_max.height)
        elif constraints.max_vram_gb < 12:
            # Use medium VRAM preset
            base_width = min(base_width, self._medium_vram_max.width)
            base_height = min(base_height, self._medium_vram_max.height)
        elif constraints.max_vram_gb < 16:
            # Use high VRAM preset
            base_width = min(base_width, self._high_vram_max.width)
            base_height = min(base_height, self._high_vram_max.height)

        # Use configured min/max values
        base_width = max(base_width, self._min_resolution.width)
        base_height = max(base_height, self._min_resolution.height)
        base_width = min(base_width, self._max_resolution.width)
        base_height = min(base_height, self._max_resolution.height)

        # Round to nearest 64
        base_width = (base_width // 64) * 64
        base_height = (base_height // 64) * 64

        return Resolution(width=base_width, height=base_height)

    def _select_sampler(
        self, analysis: PromptAnalysis, constraints: GenerationConstraints
    ) -> str:
        """
        Select optimal sampler from configuration.

        Decision matrix:
        - Priority SPEED → Fast sampler
        - Style-based selection
        - Default
        """
        # Speed priority
        if constraints.priority == Priority.SPEED:
            return "Euler A"  # Fast sampler

        # Style-based selection
        if analysis.intent:
            style = analysis.intent.artistic_style
            style_key = style.value

            if style_key == "PHOTOREALISTIC":
                return self._sdxl_sampler
            elif style_key == "ANIME":
                return self._sd20_sampler
            elif style_key == "CARTOON":
                return "Euler A"
            elif style_key == "PAINTING":
                return self._sdxl_sampler
            elif style_key == "SKETCH":
                return "Euler A"
            elif style_key == "ABSTRACT":
                return "Euler A"
            elif style_key == "CONCEPT_ART":
                return self._sd20_sampler

        # Default
        return self._sdxl_sampler

    def _determine_clip_skip(self, analysis: PromptAnalysis) -> int:
        """
        Determine CLIP skip value from configuration.
        """
        # Use default clip skip
        return self._default_clip_skip

    def _estimate_vram(self, resolution: Resolution, steps: int) -> float:
        """
        Estimate VRAM usage.

        Rough estimates for SDXL:
        - 1024x1024: ~8-10 GB
        - 768x768: ~5-6 GB
        - 512x512: ~3-4 GB
        """
        pixels = resolution.width * resolution.height

        # Base VRAM (GB) = pixels / 100000
        base_vram = pixels / 100000.0

        # Steps add minimal overhead
        step_overhead = steps * 0.01

        total_vram = base_vram + step_overhead

        return round(total_vram, 2)

    def _estimate_time(
        self, resolution: Resolution, steps: int, sampler: str
    ) -> float:
        """
        Estimate generation time.

        Assumptions:
        - NVIDIA RTX 3090 / 4090 class GPU
        - SDXL base model
        """
        pixels = resolution.width * resolution.height

        # Base time per step (seconds)
        # Assumes ~0.5s/step for 1024x1024 on RTX 4090
        base_time_per_step = (pixels / (1024 * 1024)) * 0.5

        # Sampler efficiency
        sampler_multiplier = 1.0
        if "Euler A" in sampler:
            sampler_multiplier = 0.8  # Faster
        elif "Karras" in sampler:
            sampler_multiplier = 1.1  # Slightly slower, better quality

        total_time = steps * base_time_per_step * sampler_multiplier

        return round(total_time, 1)

    def _estimate_quality(self, steps: int, cfg: float) -> float:
        """
        Estimate quality score (0-1).

        Based on:
        - Number of steps
        - CFG scale
        """
        # Steps contribution (more steps = higher quality up to a point)
        step_quality = min(steps / 50.0, 1.0)

        # CFG contribution (sweet spot around 7-9)
        cfg_quality = 1.0 - abs(cfg - 8.0) / 10.0
        cfg_quality = max(cfg_quality, 0.0)

        # Combined
        quality = 0.6 * step_quality + 0.4 * cfg_quality

        return round(quality, 2)

    def _get_strategy_name(self, constraints: GenerationConstraints) -> str:
        """Get human-readable strategy name."""
        if constraints.priority == Priority.SPEED:
            return "speed_optimized"
        elif constraints.priority == Priority.QUALITY:
            return "quality_optimized"
        else:
            return "balanced"
