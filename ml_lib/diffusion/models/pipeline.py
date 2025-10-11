"""Pipeline models and entities for intelligent image generation.

Consolidates: PipelineConfig, GenerationResult, GenerationExplanation,
Recommendations, BatchConfig, and metadata structures.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional
from PIL import Image
import json

# METADATA STRUCTURES (from metadata_dict.py)

@dataclass
class LoRASerializable:
    """Serialized LoRA information."""

    name: str
    alpha: float
    source: str


@dataclass
class GenerationMetadataSerializable:
    """Typed structure for generation metadata serialization.

    Use dataclasses.asdict() for JSON serialization.
    """

    prompt: str
    negative_prompt: str
    seed: int
    steps: int
    cfg_scale: float
    width: int
    height: int
    sampler: str
    loras_used: list[LoRASerializable]
    generation_time_seconds: float
    peak_vram_gb: float
    base_model_id: str
    pipeline_type: str

# PIPELINE CONFIGURATION (from pipeline_config.py)

class OperationMode(Enum):
    """Operation mode for the pipeline."""

    AUTO = "auto"
    """Fully automatic - all decisions made by AI."""

    ASSISTED = "assisted"
    """AI suggests, user confirms."""

    MANUAL = "manual"
    """User has full control."""


class Priority(Enum):
    """Generation priority."""

    SPEED = "speed"
    """Optimize for fastest generation."""

    QUALITY = "quality"
    """Optimize for best quality."""

    BALANCED = "balanced"
    """Balance speed and quality."""


@dataclass
class GenerationConstraints:
    """Constraints for generation."""

    max_time_seconds: Optional[float] = None
    """Maximum generation time in seconds."""

    max_vram_gb: Optional[float] = None
    """Maximum VRAM usage in GB."""

    priority: Priority = Priority.BALANCED
    """Generation priority."""

    def __post_init__(self):
        """Validate constraints."""
        if self.max_time_seconds is not None and self.max_time_seconds <= 0:
            raise ValueError("max_time_seconds must be positive")
        if self.max_vram_gb is not None and self.max_vram_gb <= 0:
            raise ValueError("max_vram_gb must be positive")


@dataclass
class LoRAPreferences:
    """LoRA selection preferences."""

    max_loras: int = 3
    """Maximum number of LoRAs to apply."""

    min_confidence: float = 0.6
    """Minimum confidence score for LoRA recommendation."""

    allow_style_mixing: bool = True
    """Allow mixing different artistic styles."""

    blocked_tags: list[str] = field(default_factory=list)
    """Tags to block from LoRA selection."""

    def __post_init__(self):
        """Validate preferences."""
        if self.max_loras < 0:
            raise ValueError("max_loras must be non-negative")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")


@dataclass
class MemorySettings:
    """Memory management settings."""

    max_vram_gb: float = 8.0
    """Maximum VRAM to use in GB."""

    offload_strategy: str = "balanced"
    """Offload strategy: 'none', 'balanced', 'aggressive'."""

    enable_quantization: bool = False
    """Enable automatic quantization."""

    quantization_dtype: str = "fp16"
    """Quantization data type: 'fp16', 'int8'."""

    def __post_init__(self):
        """Validate settings."""
        if self.max_vram_gb <= 0:
            raise ValueError("max_vram_gb must be positive")
        if self.offload_strategy not in ("none", "balanced", "aggressive"):
            raise ValueError("offload_strategy must be 'none', 'balanced', or 'aggressive'")
        if self.quantization_dtype not in ("fp16", "int8", "int4"):
            raise ValueError("quantization_dtype must be 'fp16', 'int8', or 'int4'")


@dataclass
class OllamaConfig:
    """Ollama LLM configuration."""

    base_url: str = "http://localhost:11434"
    """Ollama API base URL."""

    model: str = "llama2"
    """Model to use for analysis."""

    timeout: float = 30.0
    """Request timeout in seconds."""

    def __post_init__(self):
        """Validate config."""
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")


@dataclass
class PipelineConfig:
    """Configuration for intelligent generation pipeline."""

    base_model: str = "stabilityai/sdxl-base-1.0"
    """Base model ID (HuggingFace or CivitAI)."""

    mode: OperationMode = OperationMode.AUTO
    """Operation mode."""

    constraints: GenerationConstraints = field(default_factory=GenerationConstraints)
    """Generation constraints."""

    lora_preferences: LoRAPreferences = field(default_factory=LoRAPreferences)
    """LoRA selection preferences."""

    memory_settings: MemorySettings = field(default_factory=MemorySettings)
    """Memory management settings."""

    ollama_config: Optional[OllamaConfig] = None
    """Ollama configuration for LLM analysis (None = use rule-based fallback)."""

    enable_learning: bool = True
    """Enable learning from user feedback."""

    cache_dir: Optional[str] = None
    """Custom cache directory (None = default)."""

    def __post_init__(self):
        """Validate config."""
        if not self.base_model:
            raise ValueError("base_model cannot be empty")

# BATCH CONFIGURATION (from batch_config.py)

class VariationStrategy(Enum):
    """Strategy for generating variations in batch mode."""

    SEED_VARIATION = "seed_variation"
    """Same parameters, different seeds."""

    PARAM_VARIATION = "param_variation"
    """Vary steps, CFG, etc."""

    LORA_VARIATION = "lora_variation"
    """Try different LoRA combinations."""

    MIXED = "mixed"
    """Mix multiple strategies."""


@dataclass
class BatchConfig:
    """Configuration for batch generation."""

    num_images: int = 4
    """Number of images to generate."""

    variation_strategy: VariationStrategy = VariationStrategy.SEED_VARIATION
    """How to generate variations."""

    parallel_generation: bool = False
    """Whether to generate in parallel (requires more VRAM)."""

    base_seed: Optional[int] = None
    """Base seed for seed variation (None = random)."""

    save_individually: bool = True
    """Whether to save each image as it's generated."""

    output_dir: Optional[str] = None
    """Output directory for batch results (None = don't save)."""

    def __post_init__(self):
        """Validate config."""
        if self.num_images <= 0:
            raise ValueError("num_images must be positive")
        if self.base_seed is not None and self.base_seed < 0:
            raise ValueError("base_seed must be non-negative")

# GENERATION EXPLANATION (from generation_explanation.py)

@dataclass
class GenerationExplanation:
    """Explanation of decisions made during generation."""

    summary: str
    """High-level summary of what was done."""

    lora_reasoning: dict[str, str] = field(default_factory=dict)
    """Reasoning for each LoRA selection (lora_name -> reasoning)."""

    parameter_reasoning: dict[str, str] = field(default_factory=dict)
    """Reasoning for parameter choices (param_name -> reasoning)."""

    performance_notes: list[str] = field(default_factory=list)
    """Performance-related notes (time, VRAM, etc.)."""

    def get_full_explanation(self) -> str:
        """
        Get full formatted explanation as multi-line string.

        Returns:
            Formatted explanation text
        """
        lines = [
            "=== Generation Explanation ===",
            "",
            "Summary:",
            f"  {self.summary}",
            "",
        ]

        if self.lora_reasoning:
            lines.append("LoRA Selection:")
            for lora_name, reasoning in self.lora_reasoning.items():
                lines.append(f"  • {lora_name}: {reasoning}")
            lines.append("")

        if self.parameter_reasoning:
            lines.append("Parameter Choices:")
            for param_name, reasoning in self.parameter_reasoning.items():
                lines.append(f"  • {param_name}: {reasoning}")
            lines.append("")

        if self.performance_notes:
            lines.append("Performance:")
            for note in self.performance_notes:
                lines.append(f"  • {note}")
            lines.append("")

        return "\n".join(lines)

# RECOMMENDATIONS (from recommendations.py)

@dataclass
class Recommendations:
    """AI recommendations for assisted mode."""

    prompt_analysis: "PromptAnalysis"
    """Analysis of the prompt."""

    suggested_loras: list["LoRARecommendation"]
    """Suggested LoRAs with confidence scores."""

    suggested_params: "OptimizedParameters"
    """Suggested generation parameters."""

    explanation: str
    """High-level explanation of recommendations."""

    def get_summary(self) -> str:
        """
        Get human-readable summary of recommendations.

        Returns:
            Formatted summary string
        """
        lines = [
            "=== AI Recommendations ===",
            "",
            f"Explanation: {self.explanation}",
            "",
            "Suggested LoRAs:",
        ]

        if self.suggested_loras:
            for rec in self.suggested_loras:
                lines.append(
                    f"  • {rec.lora_name} (confidence: {rec.confidence_score:.2f}, "
                    f"alpha: {rec.suggested_alpha:.2f})"
                )
        else:
            lines.append("  (none)")

        lines.extend([
            "",
            "Suggested Parameters:",
            f"  • Steps: {self.suggested_params.num_steps}",
            f"  • CFG Scale: {self.suggested_params.guidance_scale}",
            f"  • Resolution: {self.suggested_params.width}x{self.suggested_params.height}",
            f"  • Sampler: {self.suggested_params.sampler_name}",
            "",
        ])

        return "\n".join(lines)

# GENERATION RESULT (from generation_result.py)

@dataclass
class LoRAInfo:
    """Information about a LoRA used in generation."""

    name: str
    """LoRA name/ID."""

    alpha: float
    """LoRA weight/alpha value."""

    source: str = "huggingface"
    """Source: 'huggingface', 'civitai', 'local'."""


@dataclass
class GenerationMetadata:
    """
    Complete metadata for a generation.

    Note: This class is kept for backwards compatibility.
    New code should use ImageMetadataEmbedding directly.
    """

    # Prompt
    prompt: str
    negative_prompt: str
    seed: int

    # Parameters
    steps: int
    cfg_scale: float
    width: int
    height: int
    sampler: str

    # LoRAs
    loras_used: list[LoRAInfo]

    # Resources
    generation_time_seconds: float
    peak_vram_gb: float

    # Model
    base_model_id: str
    pipeline_type: str = "intelligent"

    def to_serializable(self) -> GenerationMetadataSerializable:
        """Convert to typed serializable structure."""
        return GenerationMetadataSerializable(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            seed=self.seed,
            steps=self.steps,
            cfg_scale=self.cfg_scale,
            width=self.width,
            height=self.height,
            sampler=self.sampler,
            loras_used=[
                LoRASerializable(name=lora.name, alpha=lora.alpha, source=lora.source)
                for lora in self.loras_used
            ],
            generation_time_seconds=self.generation_time_seconds,
            peak_vram_gb=self.peak_vram_gb,
            base_model_id=self.base_model_id,
            pipeline_type=self.pipeline_type,
        )

    def to_image_metadata(
        self,
        generation_id: Optional[str] = None,
        base_model_architecture: str = "unknown",
        scheduler: str = "default",
        clip_skip: int = 2,
        vae_model: Optional[str] = None,
    ):
        """
        Convert to ImageMetadataEmbedding for enhanced metadata support.

        Args:
            generation_id: Custom generation ID (None = auto-generate)
            base_model_architecture: Architecture (SD1.5, SDXL, Flux, etc.)
            scheduler: Scheduler name
            clip_skip: CLIP skip value
            vae_model: VAE model if custom

        Returns:
            ImageMetadataEmbedding with complete metadata
        """
        # Import here to avoid circular import
        from ml_lib.diffusion.services.image_metadata import (
            ImageMetadataEmbedding,
            create_generation_id,
            create_timestamp,
        )

        return ImageMetadataEmbedding(
            generation_id=generation_id or create_generation_id(),
            generation_timestamp=create_timestamp(),
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            seed=self.seed,
            steps=self.steps,
            cfg_scale=self.cfg_scale,
            width=self.width,
            height=self.height,
            sampler=self.sampler,
            scheduler=scheduler,
            clip_skip=clip_skip,
            base_model_id=self.base_model_id,
            base_model_architecture=base_model_architecture,
            vae_model=vae_model,
            loras_used=[
                {
                    "name": lora.name,
                    "weight": lora.alpha,
                    "source": lora.source,
                }
                for lora in self.loras_used
            ],
            generation_time_seconds=self.generation_time_seconds,
            peak_vram_gb=self.peak_vram_gb,
            pipeline_type=self.pipeline_type,
        )


@dataclass
class GenerationResult:
    """Result of an image generation."""

    id: str
    """Unique generation ID."""

    image: Image.Image
    """Generated image."""

    metadata: GenerationMetadata
    """Generation metadata."""

    explanation: GenerationExplanation
    """Explanation of decisions made."""

    def save(
        self,
        path: Path | str,
        save_metadata: bool = True,
        save_explanation: bool = True,
        naming_config: Optional["ImageNamingConfig"] = None,
        use_auto_naming: bool = False,
    ):
        """
        Save image with optional metadata and explanation.

        Args:
            path: Path to save image (PNG format) or output directory if use_auto_naming=True
            save_metadata: Whether to embed metadata in PNG
            save_explanation: Whether to save explanation as JSON
            naming_config: Custom naming configuration (uses standard if None)
            use_auto_naming: If True, path is treated as output directory and filename is auto-generated

        Examples:
            >>> # Save with custom filename
            >>> result.save("/outputs/my_image.png")
            >>>
            >>> # Save with auto-generated filename (timestamp + GUID)
            >>> result.save("/outputs", use_auto_naming=True)
            >>> # Result: /outputs/20250111_143022_a3f2e9d4.png
            >>>
            >>> # Save with descriptive auto-naming
            >>> from ml_lib.diffusion.services.image_naming import ImageNamingConfig
            >>> result.save(
            ...     "/outputs",
            ...     use_auto_naming=True,
            ...     naming_config=ImageNamingConfig.descriptive()
            ... )
            >>> # Result: /outputs/20250111_143022_beautiful-sunset_a3f2e9d4.png
        """
        # Import here to avoid circular import
        from ml_lib.diffusion.services.image_metadata import (
            ImageMetadataWriter,
        )
        from ml_lib.diffusion.services.image_naming import (
            ImageNamingConfig,
        )

        path = Path(path)

        if use_auto_naming:
            # path is output directory, generate filename
            output_dir = path
            writer = ImageMetadataWriter(naming_config)
            image_metadata = self.metadata.to_image_metadata()
            filename = writer.generate_filename(image_metadata)
            path = output_dir / filename
        else:
            # path is full path to file
            path.parent.mkdir(parents=True, exist_ok=True)

        # Save image with metadata
        if save_metadata:
            # Use enhanced metadata system
            writer = ImageMetadataWriter(naming_config)
            image_metadata = self.metadata.to_image_metadata()

            # Save with embedded metadata
            saved_path = writer.save_with_metadata(
                image=self.image,
                metadata=image_metadata,
                output_dir=path.parent,
                filename=path.name,
                embed_full_json=True,
                embed_exif=True,
                save_sidecar_json=False,
            )
            path = saved_path
        else:
            # Save without metadata
            path.parent.mkdir(parents=True, exist_ok=True)
            self.image.save(path)

        # Save explanation as separate JSON
        if save_explanation:
            explanation_path = path.with_suffix(".explanation.json")
            with open(explanation_path, "w") as f:
                json.dump(
                    {
                        "summary": self.explanation.summary,
                        "lora_reasoning": self.explanation.lora_reasoning,
                        "parameter_reasoning": self.explanation.parameter_reasoning,
                        "performance_notes": self.explanation.performance_notes,
                    },
                    f,
                    indent=2,
                )

        return path

    def save_metadata_json(self, path: Path | str):
        """
        Save metadata as standalone JSON file.

        Args:
            path: Path to save JSON
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        serializable = self.metadata.to_serializable()
        with open(path, "w") as f:
            json.dump(asdict(serializable), f, indent=2)
