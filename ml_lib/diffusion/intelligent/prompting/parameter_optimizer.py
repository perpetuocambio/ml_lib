"""Parameter optimization for image generation."""

import logging
import numpy as np
from dataclasses import dataclass

from ml_lib.diffusion.intelligent.prompting.entities import (
    PromptAnalysis,
    OptimizedParameters,
    Priority,
    ArtisticStyle,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationConstraints:
    """Constraints for generation."""

    max_time_seconds: float = 120.0
    max_vram_gb: float = 16.0
    priority: Priority = Priority.BALANCED


class ParameterOptimizer:
    """Optimizes generation parameters based on prompt analysis."""

    # Sampler recommendations by style
    SAMPLER_MAP = {
        ArtisticStyle.PHOTOREALISTIC: "DPM++ 2M Karras",
        ArtisticStyle.ANIME: "DPM++ 2M",
        ArtisticStyle.CARTOON: "Euler A",
        ArtisticStyle.PAINTING: "DPM++ 2M Karras",
        ArtisticStyle.SKETCH: "Euler A",
        ArtisticStyle.ABSTRACT: "Euler A",
        ArtisticStyle.CONCEPT_ART: "DPM++ 2M",
    }

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
        Optimize number of inference steps for EXPLICIT content.

        Explicit anatomical detail requires MORE steps.

        Formula:
        steps = base + complexity_bonus + quality_bonus + anatomy_bonus - speed_penalty
        """
        # Higher base for explicit content
        base = 25

        # Complexity adjustment (INCREASED for explicit)
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

        # Clamp to EXTENDED range for explicit content (up to 80 steps)
        return int(np.clip(steps, 20, 80))

    def _optimize_cfg(self, analysis: PromptAnalysis) -> float:
        """
        Optimize CFG (guidance) scale for EXPLICIT PHOTOREALISTIC content.

        Explicit photorealistic adult content requires:
        - HIGH CFG for anatomical accuracy (genitals, positioning, detail)
        - Range: 8.0 - 15.0 (sweet spot 10.0-13.0 for explicit)
        - Higher values needed for complex anatomical descriptions
        """
        # START HIGHER for explicit content
        base_cfg = 10.5

        # Adjust for complexity (more details = MUCH higher CFG)
        if analysis.complexity_score > 0.8:
            base_cfg += 2.0  # Very complex = maximum CFG
        elif analysis.complexity_score > 0.7:
            base_cfg += 1.5
        elif analysis.complexity_score > 0.5:
            base_cfg += 1.0

        # Adjust for quality keywords (explicit needs maximum quality)
        quality_concepts = analysis.detected_concepts.get("quality", [])
        if any("detailed" in q.lower() or "hyperrealistic" in q.lower() or "8k" in q.lower()
               for q in quality_concepts):
            base_cfg += 1.0

        # CRITICAL: Anatomy focus (needs VERY precise control)
        anatomy_concepts = analysis.detected_concepts.get("anatomy", [])
        anatomy_count = len(anatomy_concepts)

        if anatomy_count > 8:
            base_cfg += 2.5  # Extremely detailed anatomy
        elif anatomy_count > 5:
            base_cfg += 2.0  # Very detailed anatomy
        elif anatomy_count > 3:
            base_cfg += 1.5  # Detailed anatomy

        # EXPLICIT: Sexual activity needs maximum CFG for positioning accuracy
        activity_concepts = analysis.detected_concepts.get("activity", [])
        activity_text = " ".join(activity_concepts).lower()

        if any(kw in activity_text for kw in ["sex", "intercourse", "penetration", "oral"]):
            base_cfg += 2.0  # Sexual acts = maximum CFG
        elif any(kw in activity_text for kw in ["intimate", "erotic", "sensual", "touching"]):
            base_cfg += 1.5  # Intimate acts = high CFG

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

        # Clamp to EXPLICIT photorealistic range (higher ceiling)
        return round(np.clip(base_cfg, 8.0, 15.0), 1)

    def _optimize_resolution(
        self, analysis: PromptAnalysis, constraints: GenerationConstraints
    ) -> tuple[int, int]:
        """
        Optimize resolution for EXPLICIT content.

        Considerations:
        - Content type (portrait vs scene)
        - Group size (single, couple, trio)
        - Anatomical detail focus (higher res for explicit)
        - VRAM constraints
        """
        # Base resolution (SDXL - HIGHER for explicit detail)
        base_width = 1024
        base_height = 1024

        # Adjust for content type and group size
        if analysis.intent:
            content = analysis.intent.content_type.value

            # Check for group scenes
            subjects = analysis.detected_concepts.get("subjects", [])
            subject_text = " ".join(subjects).lower()

            # Detect group size
            is_trio = any(kw in subject_text for kw in ["three", "trio", "group"])
            is_couple = any(kw in subject_text for kw in ["couple", "two", "duo"])

            if is_trio:
                # Trio = wider landscape for 3 people
                base_width = 1280
                base_height = 896
            elif is_couple:
                # Couple = slightly wider or taller depending on activity
                activity_concepts = analysis.detected_concepts.get("activity", [])
                if any("lying" in a or "horizontal" in a for a in activity_concepts):
                    base_width = 1216
                    base_height = 832
                else:
                    base_width = 1024
                    base_height = 1024
            elif content in ["portrait", "character"]:
                # Single portrait
                base_width = 896
                base_height = 1152
            elif content == "scene":
                # Scene (default landscape)
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

        # Adjust for VRAM constraints (may reduce from above)
        if constraints.max_vram_gb < 8:
            # Scale down for low VRAM
            base_width = int(base_width * 0.75)
            base_height = int(base_height * 0.75)
            base_width = (base_width // 64) * 64
            base_height = (base_height // 64) * 64
        elif constraints.max_vram_gb < 10:
            # Slight reduction for medium VRAM
            base_width = int(base_width * 0.875)
            base_height = int(base_height * 0.875)
            base_width = (base_width // 64) * 64
            base_height = (base_height // 64) * 64

        # Ensure minimum resolution (lower for explicit to allow detail)
        base_width = max(base_width, 768)
        base_height = max(base_height, 768)

        # Maximum resolution cap (VRAM safety)
        base_width = min(base_width, 1536)
        base_height = min(base_height, 1536)

        return (base_width, base_height)

    def _select_sampler(
        self, analysis: PromptAnalysis, constraints: GenerationConstraints
    ) -> str:
        """
        Select optimal sampler.

        Decision matrix:
        - Priority SPEED → Euler A
        - Priority QUALITY + Photorealistic → DPM++ 2M Karras
        - Priority QUALITY + Anime → DPM++ 2M
        - Balanced → DPM++ 2M Karras
        """
        # Speed priority
        if constraints.priority == Priority.SPEED:
            return "Euler A"

        # Style-based selection
        if analysis.intent:
            style = analysis.intent.artistic_style
            if style in self.SAMPLER_MAP:
                return self.SAMPLER_MAP[style]

        # Default
        return "DPM++ 2M Karras"

    def _determine_clip_skip(self, analysis: PromptAnalysis) -> int:
        """
        Determine CLIP skip value for PHOTOREALISTIC content.

        Photorealistic ALWAYS uses clip_skip=1 for maximum detail/accuracy.
        Never skip CLIP layers for realistic human anatomy.
        """
        # ALWAYS 1 for photorealistic content
        return 1

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
