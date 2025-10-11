"""Generation result entities."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from PIL import Image
import json

from .metadata_dict import GenerationMetadataSerializable, LoRASerializable


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
        from ml_lib.diffusion.intelligent.pipeline.services.image_metadata import (
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

    explanation: "GenerationExplanation"
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
            >>> from ml_lib.diffusion.intelligent.pipeline.services.image_naming import ImageNamingConfig
            >>> result.save(
            ...     "/outputs",
            ...     use_auto_naming=True,
            ...     naming_config=ImageNamingConfig.descriptive()
            ... )
            >>> # Result: /outputs/20250111_143022_beautiful-sunset_a3f2e9d4.png
        """
        # Import here to avoid circular import
        from ml_lib.diffusion.intelligent.pipeline.services.image_metadata import (
            ImageMetadataWriter,
        )
        from ml_lib.diffusion.intelligent.pipeline.services.image_naming import (
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
