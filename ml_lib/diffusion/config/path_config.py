"""
Path configuration for model discovery.

Provides flexible, OOP-based configuration for model paths without
hardcoded environment variables or assumptions.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ml_lib.diffusion.models import ModelType

logger = logging.getLogger(__name__)


@dataclass
class ModelPathConfig:
    """
    Configuration for model paths.

    This is a clean, OOP approach to configuring where models are located.
    No hardcoded paths or environment variables.

    Example:
        >>> # Option 1: Explicit paths
        >>> config = ModelPathConfig(
        ...     lora_paths=["/models/loras", "/backup/loras"],
        ...     checkpoint_paths=["/models/checkpoints"],
        ... )

        >>> # Option 2: From root directory
        >>> config = ModelPathConfig.from_root("/src/ComfyUI")
        >>> # Auto-discovers: /src/ComfyUI/models/loras, etc.

        >>> # Option 3: Empty (for programmatic addition)
        >>> config = ModelPathConfig()
        >>> config.add_model_path(ModelType.LORA, "/custom/loras")
    """

    # Per-type model paths
    checkpoint_paths: list[Path] = field(default_factory=list)
    lora_paths: list[Path] = field(default_factory=list)
    controlnet_paths: list[Path] = field(default_factory=list)
    vae_paths: list[Path] = field(default_factory=list)
    embedding_paths: list[Path] = field(default_factory=list)
    clip_paths: list[Path] = field(default_factory=list)
    clip_vision_paths: list[Path] = field(default_factory=list)
    text_encoder_paths: list[Path] = field(default_factory=list)
    unet_paths: list[Path] = field(default_factory=list)

    # Metadata
    description: str = "Model path configuration"

    def __post_init__(self):
        """Convert strings to Paths."""
        for attr in [
            "checkpoint_paths",
            "lora_paths",
            "controlnet_paths",
            "vae_paths",
            "embedding_paths",
            "clip_paths",
            "clip_vision_paths",
            "text_encoder_paths",
            "unet_paths",
        ]:
            paths = getattr(self, attr)
            setattr(self, attr, [Path(p) for p in paths])

    @classmethod
    def from_root(
        cls,
        root_dir: Path | str,
        subdirs: Optional[dict[ModelType, str]] = None,
    ) -> "ModelPathConfig":
        """
        Create configuration from a root directory.

        Args:
            root_dir: Root directory containing model subdirectories
            subdirs: Custom subdirectory mapping (None = use defaults)

        Returns:
            ModelPathConfig with discovered paths

        Example:
            >>> # ComfyUI structure
            >>> config = ModelPathConfig.from_root("/src/ComfyUI")

            >>> # Custom structure
            >>> config = ModelPathConfig.from_root(
            ...     "/my/models",
            ...     subdirs={
            ...         ModelType.LORA: "my_loras",
            ...         ModelType.VAE: "my_vaes",
            ...     }
            ... )
        """
        root = Path(root_dir)

        # Default subdirectory structure (ComfyUI-compatible)
        default_subdirs = {
            ModelType.BASE_MODEL: "models/checkpoints",
            ModelType.LORA: "models/loras",
            ModelType.CONTROLNET: "models/controlnet",
            ModelType.VAE: "models/vae",
            ModelType.EMBEDDING: "models/embeddings",
            ModelType.CLIP: "models/clip",
            ModelType.CLIP_VISION: "models/clip_vision",
            ModelType.TEXT_ENCODER: "models/text_encoders",
            ModelType.UNET: "models/unet",
        }

        # Use custom subdirs if provided
        subdirs_to_use = subdirs or default_subdirs

        # Build path lists
        checkpoint_paths = []
        lora_paths = []
        controlnet_paths = []
        vae_paths = []
        embedding_paths = []
        clip_paths = []
        clip_vision_paths = []
        text_encoder_paths = []
        unet_paths = []

        for model_type, subdir in subdirs_to_use.items():
            full_path = root / subdir

            # Resolve symlinks
            if full_path.is_symlink():
                full_path = full_path.resolve()

            if full_path.exists():
                if model_type == ModelType.BASE_MODEL:
                    checkpoint_paths.append(full_path)
                elif model_type == ModelType.LORA:
                    lora_paths.append(full_path)
                elif model_type == ModelType.CONTROLNET:
                    controlnet_paths.append(full_path)
                elif model_type == ModelType.VAE:
                    vae_paths.append(full_path)
                elif model_type == ModelType.EMBEDDING:
                    embedding_paths.append(full_path)
                elif model_type == ModelType.CLIP:
                    clip_paths.append(full_path)
                elif model_type == ModelType.CLIP_VISION:
                    clip_vision_paths.append(full_path)
                elif model_type == ModelType.TEXT_ENCODER:
                    text_encoder_paths.append(full_path)
                elif model_type == ModelType.UNET:
                    unet_paths.append(full_path)

        return cls(
            checkpoint_paths=checkpoint_paths,
            lora_paths=lora_paths,
            controlnet_paths=controlnet_paths,
            vae_paths=vae_paths,
            embedding_paths=embedding_paths,
            clip_paths=clip_paths,
            clip_vision_paths=clip_vision_paths,
            text_encoder_paths=text_encoder_paths,
            unet_paths=unet_paths,
            description=f"Discovered from {root}",
        )

    def get_paths(self, model_type: ModelType) -> list[Path]:
        """
        Get paths for a specific model type.

        Args:
            model_type: Type of model

        Returns:
            List of paths for that model type
        """
        mapping = {
            ModelType.BASE_MODEL: self.checkpoint_paths,
            ModelType.LORA: self.lora_paths,
            ModelType.CONTROLNET: self.controlnet_paths,
            ModelType.VAE: self.vae_paths,
            ModelType.EMBEDDING: self.embedding_paths,
            ModelType.CLIP: self.clip_paths,
            ModelType.CLIP_VISION: self.clip_vision_paths,
            ModelType.TEXT_ENCODER: self.text_encoder_paths,
            ModelType.UNET: self.unet_paths,
        }
        return mapping.get(model_type, [])

    def add_model_path(self, model_type: ModelType, path: Path | str) -> None:
        """
        Add a path for a model type.

        Args:
            model_type: Type of model
            path: Path to add
        """
        path = Path(path)
        paths = self.get_paths(model_type)
        if path not in paths:
            paths.append(path)
            logger.info(f"Added {model_type.value} path: {path}")

    def has_paths(self, model_type: ModelType) -> bool:
        """Check if any paths are configured for model type."""
        return len(self.get_paths(model_type)) > 0

    def get_all_configured_types(self) -> list[ModelType]:
        """Get list of model types that have paths configured."""
        configured = []
        for model_type in ModelType:
            if self.has_paths(model_type):
                configured.append(model_type)
        return configured

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "checkpoints": [str(p) for p in self.checkpoint_paths],
            "loras": [str(p) for p in self.lora_paths],
            "controlnets": [str(p) for p in self.controlnet_paths],
            "vaes": [str(p) for p in self.vae_paths],
            "embeddings": [str(p) for p in self.embedding_paths],
            "clip": [str(p) for p in self.clip_paths],
            "clip_vision": [str(p) for p in self.clip_vision_paths],
            "text_encoders": [str(p) for p in self.text_encoder_paths],
            "unets": [str(p) for p in self.unet_paths],
            "description": self.description,
        }
