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
from ml_lib.diffusion.handlers.config_loader import get_default_config

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
            config: PrompterConfig with configuration (if None, loads default)
        """
        # Load configuration
        if config is None:
            config = get_default_config()
        self.config = config
        
        # Set up configurable values
        self.SAMPLER_MAP = self.config.model_strategies
        self.DEFAULT_RANGES = self.config.default_ranges
        self.VRAM_PRESETS = self.config.vram_presets
        self.ACTIVITY_PROFILES = self.config.activity_profiles
        self.AGE_PROFILES = self.config.age_profiles
        self.DETAIL_PRESETS = self.config.detail_presets

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
            width=resolution[0],
            height=resolution[1],
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
        # Use base from configuration, defaulting to 25
        base = self.DETAIL_PRESETS.get("medium", {}).get("base_steps", 35)

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

        # Use configurable range
        min_steps = self.DEFAULT_RANGES.get("min_steps", 20)
        max_steps = self.DEFAULT_RANGES.get("max_steps", 80)
        
        return int(np.clip(steps, min_steps, max_steps))

    def _optimize_cfg(self, analysis: PromptAnalysis) -> float:
        """
        Optimize CFG (guidance) scale from configuration.

        Uses configurable ranges and base values.
        """
        # START with configurable base
        base_cfg = self.DETAIL_PRESETS.get("medium", {}).get("base_cfg", 9.0)

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

        # Use configurable range
        min_cfg = self.DEFAULT_RANGES.get("min_cfg", 7.0)
        max_cfg = self.DEFAULT_RANGES.get("max_cfg", 15.0)
        
        return round(np.clip(base_cfg, min_cfg, max_cfg), 1)

    def _optimize_resolution(
        self, analysis: PromptAnalysis, constraints: GenerationConstraints
    ) -> tuple[int, int]:
        """
        Optimize resolution from configuration.

        Considerations:
        - Content type (portrait vs scene)
        - Group size (single, couple, trio)
        - Anatomical detail focus
        - VRAM constraints
        """
        # Use configurable base resolution or defaults
        base_width, base_height = self.DETAIL_PRESETS.get("medium", {}).get("base_resolution", [1024, 1024])

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
                # Trio - use configuration or default
                base_width, base_height = self.config.group_profiles.get("trio", {}).get("default_resolution", [1280, 896])
            elif is_couple:
                # Couple - use configuration or default
                base_width, base_height = self.config.group_profiles.get("couple", {}).get("default_resolution", [1024, 1024])
                # Adjust for activity
                activity_concepts = analysis.detected_concepts.get("activity", [])
                if any("lying" in a or "horizontal" in a for a in activity_concepts):
                    base_width, base_height = [1216, 832]
            elif content in ["portrait", "character"]:
                # Single portrait - use configuration or default
                base_width, base_height = [896, 1152]
            elif content == "scene":
                # Scene - use configuration or default
                base_width, base_height = [1152, 896]

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

        # Adjust for VRAM constraints using configuration presets
        if constraints.max_vram_gb < 8:
            # Use low VRAM preset
            preset_res = self.VRAM_PRESETS.get("low_vram", {}).get("max_resolution", [768, 768])
            base_width = min(base_width, preset_res[0])
            base_height = min(base_height, preset_res[1])
        elif constraints.max_vram_gb < 12:
            # Use medium VRAM preset
            preset_res = self.VRAM_PRESETS.get("medium_vram", {}).get("max_resolution", [1024, 1024])
            base_width = min(base_width, preset_res[0])
            base_height = min(base_height, preset_res[1])
        elif constraints.max_vram_gb < 16:
            # Use high VRAM preset
            preset_res = self.VRAM_PRESETS.get("high_vram", {}).get("max_resolution", [1216, 832])
            base_width = min(base_width, preset_res[0])
            base_height = min(base_height, preset_res[1])

        # Use configurable min/max values
        min_res = self.DEFAULT_RANGES.get("min_resolution", [768, 768])
        max_res = self.DEFAULT_RANGES.get("max_resolution", [1536, 1536])

        base_width = max(base_width, min_res[0])
        base_height = max(base_height, min_res[1])
        base_width = min(base_width, max_res[0])
        base_height = min(base_height, max_res[1])

        # Round to nearest 64
        base_width = (base_width // 64) * 64
        base_height = (base_height // 64) * 64

        return (base_width, base_height)

    def _select_sampler(
        self, analysis: PromptAnalysis, constraints: GenerationConstraints
    ) -> str:
        """
        Select optimal sampler from configuration.

        Decision matrix:
        - Priority SPEED → Configurable speed sampler
        - Style-based selection from configuration
        - Default from configuration
        """
        # Speed priority
        if constraints.priority == Priority.SPEED:
            # Use a fast sampler from the model strategies or default to "Euler A"
            return self.SAMPLER_MAP.get("sdxl", {}).get("default_sampler", "Euler A")

        # Style-based selection
        if analysis.intent:
            style = analysis.intent.artistic_style
            # Use the style name as key, fallback to lowercase name
            style_key = style.value  # This gives us the string value like "photorealistic"
            
            # Look for a matching sampler in the configuration
            if style_key in self.SAMPLER_MAP:
                sampler = self.SAMPLER_MAP[style_key].get("default_sampler")
                if sampler:
                    return sampler
            elif style_key == "PHOTOREALISTIC":
                return self.SAMPLER_MAP.get("sdxl", {}).get("default_sampler", "DPM++ 2M Karras")
            elif style_key == "ANIME":
                return self.SAMPLER_MAP.get("sd20", {}).get("default_sampler", "DPM++ 2M")
            elif style_key == "CARTOON":
                return self.SAMPLER_MAP.get("sd20", {}).get("default_sampler", "Euler A")
            elif style_key == "PAINTING":
                return self.SAMPLER_MAP.get("sdxl", {}).get("default_sampler", "DPM++ 2M Karras")
            elif style_key == "SKETCH":
                return self.SAMPLER_MAP.get("sd20", {}).get("default_sampler", "Euler A")
            elif style_key == "ABSTRACT":
                return self.SAMPLER_MAP.get("sd20", {}).get("default_sampler", "Euler A")
            elif style_key == "CONCEPT_ART":
                return self.SAMPLER_MAP.get("sd20", {}).get("default_sampler", "DPM++ 2M")

        # Default from config or fallback
        return self.SAMPLER_MAP.get("sdxl", {}).get("default_sampler", "DPM++ 2M Karras")

    def _determine_clip_skip(self, analysis: PromptAnalysis) -> int:
        """
        Determine CLIP skip value from configuration.
        """
        # Use default from config, defaulting to 1 for photorealistic content
        if analysis.intent and analysis.intent.artistic_style == ArtisticStyle.PHOTOREALISTIC:
            return self.SAMPLER_MAP.get("sdxl", {}).get("default_clip_skip", 1)
        else:
            return self.SAMPLER_MAP.get("sdxl", {}).get("default_clip_skip", 1)

    def _estimate_vram(self, resolution: tuple[int, int], steps: int) -> float:
        """
        Estimate VRAM usage.

        Rough estimates for SDXL:
        - 1024x1024: ~8-10 GB
        - 768x768: ~5-6 GB
        - 512x512: ~3-4 GB
        """
        width, height = resolution
        pixels = width * height

        # Base VRAM (GB) = pixels / 100000
        base_vram = pixels / 100000.0

        # Steps add minimal overhead
        step_overhead = steps * 0.01

        total_vram = base_vram + step_overhead

        return round(total_vram, 2)

    def _estimate_time(
        self, resolution: tuple[int, int], steps: int, sampler: str
    ) -> float:
        """
        Estimate generation time.

        Assumptions:
        - NVIDIA RTX 3090 / 4090 class GPU
        - SDXL base model
        """
        width, height = resolution
        pixels = width * height

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
