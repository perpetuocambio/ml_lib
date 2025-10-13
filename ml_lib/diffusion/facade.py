"""
Simplified facade for the diffusion module.

This provides a clean, simple interface that hides the complexity of the
underlying intelligent generation system.

Example:
    Basic usage:
    >>> from ml_lib.diffusion import ImageGenerator
    >>> generator = ImageGenerator()
    >>> image = generator.generate_from_prompt("a beautiful woman")
    >>> image.save("woman.png")

    Advanced usage:
    >>> generator = ImageGenerator(model="stabilityai/stable-diffusion-xl-base-1.0")
    >>> image = generator.generate_from_prompt(
    ...     "a beautiful woman with blue eyes",
    ...     negative_prompt="low quality",
    ...     steps=50,
    ...     seed=42
    ... )
    >>> image.save("custom.png")
"""

from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass
from PIL import Image

from ml_lib.diffusion.services import IntelligentPipelineBuilder
from ml_lib.diffusion.models.pipeline import (
    PipelineConfig,
    MemorySettings,
    OffloadStrategy,
    LoRAPreferences,
)
from ml_lib.diffusion.models.value_objects import PromptAnalysisResult


@dataclass
class GenerationOptions:
    """Options for image generation."""

    # Prompt settings
    negative_prompt: str = "low quality, blurry, deformed, bad anatomy"

    # Quality settings
    steps: int = 35
    cfg_scale: float = 7.5

    # Size settings
    width: int = 1024
    height: int = 1024

    # Reproducibility
    seed: Optional[int] = None

    # Memory management
    memory_mode: Literal["auto", "low", "balanced", "aggressive"] = "auto"

    # Advanced features
    enable_loras: bool = True
    enable_learning: bool = False


class ImageGenerator:
    """
    Simplified facade for diffusion-based image generation.

    This class provides a clean interface to the complex intelligent generation
    pipeline, hiding implementation details and making it easy to generate images.

    Attributes:
        model: The base model identifier to use
        device: Device to run on ("cuda", "cpu", or "auto")
        options: Default generation options
    """

    def __init__(
        self,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: Literal["cuda", "cpu", "auto"] = "auto",
        cache_dir: Optional[Path] = None,
        options: Optional[GenerationOptions] = None
    ):
        """
        Initialize the image generator.

        Args:
            model: Base diffusion model to use
            device: Device to run on ("cuda", "cpu", or "auto")
            cache_dir: Directory for caching models and data
            options: Default generation options
        """
        self.model = model
        self.device = device
        self.cache_dir = cache_dir or Path.home() / ".cache" / "ml_lib" / "diffusion"
        self.options = options or GenerationOptions()

        # Lazy initialization - will be set on first use
        self._pipeline = None

    def _init_pipeline(self) -> None:
        """Initialize the intelligent generation pipeline (lazy)."""
        if self._pipeline is not None:
            return

        try:
            # Map simple memory mode to complex config using explicit logic
            if self.options.memory_mode == "auto":
                offload_strategy = OffloadStrategy.BALANCED
            elif self.options.memory_mode == "low":
                offload_strategy = OffloadStrategy.AGGRESSIVE
            elif self.options.memory_mode == "balanced":
                offload_strategy = OffloadStrategy.BALANCED
            elif self.options.memory_mode == "aggressive":
                offload_strategy = OffloadStrategy.AGGRESSIVE
            else:
                offload_strategy = OffloadStrategy.BALANCED

            memory_settings = MemorySettings(
                offload_strategy=offload_strategy,
                enable_quantization=True,
                enable_vae_tiling=True,
                max_vram_gb=None  # Auto-detect
            )

            lora_prefs = LoRAPreferences(
                max_loras=5 if self.options.enable_loras else 0,
                min_confidence=0.7,
                prefer_local=True
            )

            config = PipelineConfig(
                base_model=self.model,
                memory_settings=memory_settings,
                lora_preferences=lora_prefs,
                enable_learning=self.options.enable_learning,
                cache_dir=self.cache_dir
            )

            # Try ComfyUI first, fallback to direct diffusers
            try:
                self._pipeline = IntelligentPipelineBuilder.from_comfyui_auto(config)
            except Exception:
                self._pipeline = IntelligentPipelineBuilder.from_diffusers(config)

        except ImportError as e:
            raise RuntimeError(
                f"Failed to initialize diffusion pipeline: {e}\n"
                "Make sure you have installed the required dependencies:\n"
                "  pip install diffusers torch transformers"
            )

    def generate_from_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate
            negative_prompt: Negative prompt to guide generation away from
            steps: Number of denoising steps (overrides default)
            cfg_scale: Classifier-free guidance scale (overrides default)
            width: Image width in pixels (overrides default)
            height: Image height in pixels (overrides default)
            seed: Random seed for reproducibility (overrides default)

        Returns:
            PIL Image of the generated image

        Example:
            >>> generator = ImageGenerator()
            >>> image = generator.generate_from_prompt(
            ...     "a beautiful sunset over mountains",
            ...     steps=40,
            ...     seed=123
            ... )
            >>> image.save("sunset.png")
        """
        self._init_pipeline()

        return self._generate_internal(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt is not None else self.options.negative_prompt,
            steps=steps if steps is not None else self.options.steps,
            cfg_scale=cfg_scale if cfg_scale is not None else self.options.cfg_scale,
            width=width if width is not None else self.options.width,
            height=height if height is not None else self.options.height,
            seed=seed if seed is not None else self.options.seed
        )

    def _generate_internal(
        self,
        prompt: str,
        negative_prompt: str,
        steps: int,
        cfg_scale: float,
        width: int,
        height: int,
        seed: Optional[int]
    ) -> Image.Image:
        """Internal method to perform generation using the pipeline."""
        result = self._pipeline.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            # Overrides for the optimizer
            num_steps=steps,
            guidance_scale=cfg_scale,
            width=width,
            height=height
        )

        return result.image

    def analyze_prompt(self, prompt: str) -> PromptAnalysisResult:
        """
        Analyze a prompt and get recommendations without generating.

        Useful for understanding what the system will do before generation.

        Args:
            prompt: Text prompt to analyze

        Returns:
            PromptAnalysisResult with concepts, emphases, and reasoning

        Example:
            >>> generator = ImageGenerator()
            >>> analysis = generator.analyze_prompt("anime girl with magic")
            >>> print(analysis.concepts)
            >>> print(analysis.emphases)
        """
        self._init_pipeline()

        recommendations = self._pipeline.analyze_and_recommend(prompt)

        # Convert PromptAnalysis to PromptAnalysisResult
        prompt_analysis = recommendations.prompt_analysis

        # Convert detected_concepts (dict[str, list[str]]) to ConceptMap
        from ml_lib.diffusion.models.value_objects import (
            ConceptMap,
            EmphasisMap,
            ReasoningMap,
            Concept,
            Emphasis,
            Reasoning,
        )

        # Build concepts from detected_concepts
        concepts_list = []
        for category, concept_names in prompt_analysis.detected_concepts.items():
            for concept_name in concept_names:
                concepts_list.append(
                    Concept(
                        name=concept_name,
                        category=category,
                        confidence=0.8  # Default confidence (PromptAnalysis doesn't track this)
                    )
                )

        concept_map = ConceptMap(concepts_list)

        # Build emphases from emphasis_map
        emphasis_list = []
        for keyword, weight in prompt_analysis.emphasis_map.items():
            emphasis_list.append(
                Emphasis(
                    keyword=keyword,
                    weight=weight,
                    position=0  # PromptAnalysis doesn't track position
                )
            )

        emphasis_map = EmphasisMap(emphasis_list)

        # Build reasoning map
        reasoning_list = []
        if hasattr(recommendations, 'explanation') and recommendations.explanation:
            reasoning_list.append(
                Reasoning(
                    decision="prompt_analysis",
                    reason=recommendations.explanation,
                    confidence=0.85
                )
            )

        # Add intent reasoning if available
        if prompt_analysis.intent:
            reasoning_list.append(
                Reasoning(
                    decision="intent_detection",
                    reason=f"Detected style: {prompt_analysis.intent.artistic_style.value}, "
                           f"Content: {prompt_analysis.intent.content_type.value}",
                    confidence=prompt_analysis.intent.confidence
                )
            )

        reasoning_map = ReasoningMap(reasoning_list)

        return PromptAnalysisResult(
            concepts=concept_map,
            emphases=emphasis_map,
            reasoning=reasoning_map
        )

    def provide_feedback(
        self,
        generation_id: str,
        rating: int,
        comments: str = ""
    ) -> None:
        """
        Provide feedback on a generation to improve future results.

        Note: Requires enable_learning=True in options.

        Args:
            generation_id: ID of the generation (from result metadata)
            rating: Rating from 1-5 (1=poor, 5=excellent)
            comments: Optional feedback comments

        Example:
            >>> generator = ImageGenerator(options=GenerationOptions(enable_learning=True))
            >>> image = generator.generate_from_prompt("a beautiful landscape")
            >>> # After reviewing the image...
            >>> generator.provide_feedback(result.id, rating=4, comments="Good but too bright")
        """
        if not self.options.enable_learning:
            raise RuntimeError(
                "Learning is not enabled. Set enable_learning=True in GenerationOptions."
            )

        self._init_pipeline()
        self._pipeline.provide_feedback(
            generation_id=generation_id,
            rating=rating,
            comments=comments
        )


# Convenience alias
Generator = ImageGenerator


__all__ = ["ImageGenerator", "Generator", "GenerationOptions"]
