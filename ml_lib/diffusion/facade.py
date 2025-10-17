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

from ml_lib.diffusion.domain.services import IntelligentPipelineBuilder
from ml_lib.diffusion.domain.value_objects_models.pipeline import (
    PipelineConfig,
    MemorySettings,
    LoRAPreferences,
)
from ml_lib.diffusion.domain.value_objects_models.memory import OffloadStrategy
from ml_lib.diffusion.domain.value_objects_models.value_objects import PromptAnalysisResult


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
        options: Optional[GenerationOptions] = None,
        ollama_model: str = "dolphin3",  # Activado por defecto
        ollama_url: Optional[str] = None,
        enable_ollama: bool = True,  # Activado por defecto
    ):
        """
        Initialize the image generator.

        Args:
            model: Base diffusion model to use
            device: Device to run on ("cuda", "cpu", or "auto")
            cache_dir: Directory for caching models and data
            options: Default generation options
            ollama_model: Name of Ollama model for intelligent selection (default: dolphin3)
            ollama_url: URL of the Ollama server
            enable_ollama: Enable Ollama for intelligent model selection (default: True)
        """
        self.model = model
        self.device = device
        self.cache_dir = cache_dir or Path.home() / ".cache" / "ml_lib" / "diffusion"
        self.options = options or GenerationOptions()
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.enable_ollama = enable_ollama

        # Lazy initialization - will be set on first use
        self._pipeline = None

    def _init_pipeline(self) -> None:
        """Initialize the intelligent generation pipeline (lazy)."""
        if self._pipeline is not None:
            return

        try:
            # The IntelligentPipelineBuilder is the pipeline now.
            # It handles all the complex setup.
            self._pipeline = IntelligentPipelineBuilder.from_comfyui_auto(
                enable_ollama=self.enable_ollama,
                ollama_model=self.ollama_model,
                ollama_url=self.ollama_url,
                device=self.device,
                enable_auto_download=True,  # Assuming auto-download is desired
            )

        except ImportError as e:
            raise RuntimeError(
                f"Failed to initialize diffusion pipeline: {e}\n"
                "Make sure you have installed the required dependencies:\n"
                "  pip install diffusers torch transformers peft"
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
        """
        self._init_pipeline()

        # Map facade arguments to the builder's generate method arguments
        # This is an approximation.
        quality = "balanced"
        if steps:
            if steps > 40:
                quality = "high"
            elif steps < 25:
                quality = "fast"

        # The builder's generate method returns a list or a single image
        result = self._pipeline.generate(
            prompt=prompt,
            negative_prompt=negative_prompt or self.options.negative_prompt,
            quality=quality,
            memory_mode=self.options.memory_mode,
            width=width or self.options.width,
            height=height or self.options.height,
            seed=seed or self.options.seed,
            # Pass through advanced overrides
            steps=steps,
            cfg_scale=cfg_scale,
        )

        # Ensure we return a single image, as per the facade's type hint
        if isinstance(result, list):
            if not result:
                raise RuntimeError("Image generation failed and returned no images.")
            return result[0]
        return result

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
        if not self.enable_ollama:
            raise RuntimeError(
                "Prompt analysis requires Ollama. "
                "Initialize ImageGenerator with ollama_model parameter."
            )

        # Use OllamaModelSelector directly for analysis
        from ml_lib.diffusion.domain.services.ollama_selector import OllamaModelSelector
        from ml_lib.diffusion.domain.value_objects_models.value_objects import (
            ConceptMap,
            EmphasisMap,
            ReasoningMap,
            Concept,
            Emphasis,
            ReasoningEntry,
        )

        selector = OllamaModelSelector(
            ollama_model=self.ollama_model,
            ollama_url=self.ollama_url
        )

        try:
            prompt_analysis = selector.analyze_prompt(prompt)
        except Exception as e:
            # Return empty result on error
            return PromptAnalysisResult(
                concepts=ConceptMap([]),
                emphases=EmphasisMap([]),
                reasoning=ReasoningMap([
                    ReasoningEntry(
                        key="analysis_error",
                        explanation=f"Failed to analyze prompt: {e}"
                    )
                ])
            )

        # Build concepts from detected_concepts
        concepts_list = []
        for category, concept_names in prompt_analysis.detected_concepts.items():
            for concept_name in concept_names:
                concepts_list.append(
                    Concept(
                        name=concept_name,
                        category=category,
                        confidence=0.8,
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
                    position=0,
                )
            )

        emphasis_map = EmphasisMap(emphasis_list)

        # Build reasoning map
        reasoning_list = []

        # Add intent reasoning if available
        if prompt_analysis.intent:
            reasoning_list.append(
                ReasoningEntry(
                    key="intent_detection",
                    explanation=f"Detected style: {prompt_analysis.intent.artistic_style.value}, "
                    f"Content: {prompt_analysis.intent.content_type.value} "
                    f"(confidence: {prompt_analysis.intent.confidence:.2f})"
                )
            )

        # Add suggested model reasoning
        if prompt_analysis.suggested_base_model:
            reasoning_list.append(
                ReasoningEntry(
                    key="model_suggestion",
                    explanation=f"Recommended architecture: {prompt_analysis.suggested_base_model}"
                )
            )

        # Ensure reasoning_list is not empty (ReasoningMap requires at least one entry)
        if not reasoning_list:
            reasoning_list.append(
                ReasoningEntry(
                    key="analysis_complete",
                    explanation="Prompt analyzed successfully via Ollama"
                )
            )

        reasoning_map = ReasoningMap(reasoning_list)

        return PromptAnalysisResult(
            concepts=concept_map,
            emphases=emphasis_map,
            reasoning=reasoning_map
        )

    def provide_feedback(
        self, generation_id: str, rating: int, comments: str = ""
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
            generation_id=generation_id, rating=rating, comments=comments
        )


# Convenience alias
Generator = ImageGenerator


__all__ = ["ImageGenerator", "Generator", "GenerationOptions"]
