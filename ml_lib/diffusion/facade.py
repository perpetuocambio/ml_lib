"""
Simplified facade for the diffusion module.

This provides a clean, simple interface that hides the complexity of the
underlying intelligent generation system.

Example:
    Basic usage:
    >>> from ml_lib.diffusion import ImageGenerator
    >>> generator = ImageGenerator()
    >>> image = generator.generate_character()
    >>> image.save("character.png")

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
        self._character_generator = None

    def _init_pipeline(self):
        """Initialize the intelligent generation pipeline (lazy)."""
        if self._pipeline is not None:
            return

        # Import here to avoid circular dependencies and allow usage without
        # full installation (for documentation, testing structure, etc.)
        try:
            from ml_lib.diffusion.services.intelligent_builder import (
                IntelligentPipelineBuilder
            )
            from ml_lib.diffusion.intelligent.pipeline.entities import (
                PipelineConfig,
                MemorySettings,
                OffloadStrategy,
                LoRAPreferences
            )

            # Map simple memory mode to complex config
            memory_strategy_map = {
                "auto": OffloadStrategy.BALANCED,
                "low": OffloadStrategy.AGGRESSIVE,
                "balanced": OffloadStrategy.BALANCED,
                "aggressive": OffloadStrategy.AGGRESSIVE,
            }

            memory_settings = MemorySettings(
                offload_strategy=memory_strategy_map[self.options.memory_mode],
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

    def _init_character_generator(self):
        """Initialize the character generator (lazy)."""
        if self._character_generator is not None:
            return

        try:
            from ml_lib.diffusion.intelligent.prompting import CharacterGenerator
            self._character_generator = CharacterGenerator()
        except ImportError as e:
            raise RuntimeError(
                f"Failed to initialize character generator: {e}\n"
                "Character generation requires the intelligent prompting module."
            )

    def generate_character(
        self,
        age_range: Optional[str] = None,
        ethnicity: Optional[str] = None,
        style: Optional[str] = None,
        **options
    ) -> Image.Image:
        """
        Generate a character image with intelligent attribute selection.

        Args:
            age_range: Optional age range (e.g., "young_adult", "milf")
            ethnicity: Optional ethnicity (e.g., "caucasian", "asian")
            style: Optional artistic style (e.g., "realistic", "anime")
            **options: Override default GenerationOptions (steps, cfg_scale, etc.)

        Returns:
            PIL Image of the generated character

        Example:
            >>> generator = ImageGenerator()
            >>> image = generator.generate_character(
            ...     age_range="young_adult",
            ...     ethnicity="asian",
            ...     steps=50
            ... )
            >>> image.save("asian_character.png")
        """
        # Initialize subsystems
        self._init_character_generator()
        self._init_pipeline()

        # Generate character
        character = self._character_generator.generate_character()
        prompt = character.to_prompt()

        # Merge options
        gen_options = self._merge_options(**options)

        # Generate image
        return self._generate_internal(
            prompt=prompt,
            negative_prompt=gen_options.negative_prompt,
            steps=gen_options.steps,
            cfg_scale=gen_options.cfg_scale,
            width=gen_options.width,
            height=gen_options.height,
            seed=gen_options.seed
        )

    def generate_from_prompt(
        self,
        prompt: str,
        **options
    ) -> Image.Image:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate
            **options: Override default GenerationOptions

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

        gen_options = self._merge_options(**options)

        return self._generate_internal(
            prompt=prompt,
            negative_prompt=gen_options.negative_prompt,
            steps=gen_options.steps,
            cfg_scale=gen_options.cfg_scale,
            width=gen_options.width,
            height=gen_options.height,
            seed=gen_options.seed
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

    def _merge_options(self, **overrides) -> GenerationOptions:
        """Merge default options with overrides."""
        # Start with defaults
        merged = GenerationOptions(
            negative_prompt=self.options.negative_prompt,
            steps=self.options.steps,
            cfg_scale=self.options.cfg_scale,
            width=self.options.width,
            height=self.options.height,
            seed=self.options.seed,
            memory_mode=self.options.memory_mode,
            enable_loras=self.options.enable_loras,
            enable_learning=self.options.enable_learning
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(merged, key):
                setattr(merged, key, value)

        return merged

    def analyze_prompt(self, prompt: str) -> dict:
        """
        Analyze a prompt and get recommendations without generating.

        Useful for understanding what the system will do before generation.

        Args:
            prompt: Text prompt to analyze

        Returns:
            Dictionary with analysis, recommended LoRAs, and parameters

        Example:
            >>> generator = ImageGenerator()
            >>> analysis = generator.analyze_prompt("anime girl with magic")
            >>> print(analysis["suggested_loras"])
            >>> print(analysis["suggested_params"])
        """
        self._init_pipeline()

        recommendations = self._pipeline.analyze_and_recommend(prompt)

        return {
            "analysis": recommendations.prompt_analysis,
            "suggested_loras": recommendations.suggested_loras,
            "suggested_params": recommendations.suggested_params,
            "explanation": recommendations.explanation
        }

    def provide_feedback(
        self,
        generation_id: str,
        rating: int,
        comments: str = ""
    ):
        """
        Provide feedback on a generation to improve future results.

        Note: Requires enable_learning=True in options.

        Args:
            generation_id: ID of the generation (from result metadata)
            rating: Rating from 1-5 (1=poor, 5=excellent)
            comments: Optional feedback comments

        Example:
            >>> generator = ImageGenerator(options=GenerationOptions(enable_learning=True))
            >>> image = generator.generate_character()
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
