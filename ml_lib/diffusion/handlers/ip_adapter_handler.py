"""IP-Adapter service for image-based conditioning with real CLIP Vision."""

import logging
from pathlib import Path
from typing import Optional, Any, Union
import numpy as np
from PIL import Image

from ml_lib.diffusion.intelligent.ip_adapter.entities import (
    IPAdapterConfig,
    ReferenceImage,
    ImageFeatures,
)

logger = logging.getLogger(__name__)


class IPAdapterService:
    """
    Main service for IP-Adapter functionality with CLIP Vision integration.

    Example:
        >>> service = IPAdapterService()
        >>> service.load_clip_vision()  # Auto-detect from ComfyUI
        >>> image = Image.open("reference.png")
        >>> ref = ReferenceImage(image=np.array(image), scale=0.8)
        >>> features = service.extract_features(ref)
    """

    def __init__(
        self,
        model_registry: Optional[Any] = None,
        clip_vision_path: Optional[str] = None,
    ):
        """
        Initialize IP-Adapter service.

        Args:
            model_registry: Model registry for IP-Adapter models
            clip_vision_path: Path to CLIP Vision model (None = auto-detect)
        """
        self.model_registry = model_registry
        self.loaded_models: dict[str, Any] = {}
        self.clip_vision_encoder: Optional[Any] = None
        self.clip_vision_path = clip_vision_path
        logger.info("IPAdapterService initialized")

    def load_clip_vision(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        auto_detect: bool = True,
    ) -> None:
        """
        Load CLIP Vision encoder for feature extraction.

        Args:
            model_path: Path to CLIP Vision model (None = use init path or auto-detect)
            device: Device to load on
            auto_detect: Auto-detect from ComfyUI if no path provided

        Example:
            >>> service.load_clip_vision()  # Auto-detect
            >>> # Or explicit:
            >>> service.load_clip_vision("/path/to/clip_vision_g.safetensors")
        """
        try:
            from ml_lib.diffusion.handlers.clip_vision_handler import load_clip_vision

            path_to_use = model_path or self.clip_vision_path

            logger.info("Loading CLIP Vision encoder...")
            self.clip_vision_encoder = load_clip_vision(
                model_path=path_to_use,
                device=device,
                auto_detect_comfyui=auto_detect,
            )
            logger.info("✅ CLIP Vision encoder loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load CLIP Vision: {e}")
            logger.warning("IP-Adapter will use placeholder features")
            self.clip_vision_encoder = None

    def load_ip_adapter(
        self, config: IPAdapterConfig, device: str = "cuda"
    ) -> None:
        """
        Load IP-Adapter model.

        Args:
            config: IP-Adapter configuration
            device: Device to load on

        Note:
            This is still a placeholder. Real IP-Adapter loading requires
            the actual IP-Adapter weights from HuggingFace or ComfyUI.
        """
        logger.info(f"Loading IP-Adapter: {config.model_id} ({config.variant.value})")
        self.loaded_models[config.model_id] = {"config": config, "model": None}
        logger.warning("IP-Adapter model loading is placeholder - weights needed")

    def extract_features(
        self,
        ref_image: Union[ReferenceImage, Image.Image, np.ndarray],
        return_patch_features: bool = True,
    ) -> ImageFeatures:
        """
        Extract features from reference image using CLIP Vision.

        Args:
            ref_image: Reference image (ReferenceImage, PIL, or numpy)
            return_patch_features: Whether to return patch-level features

        Returns:
            ImageFeatures with global and optionally patch features

        Example:
            >>> image = Image.open("reference.png")
            >>> features = service.extract_features(image)
            >>> print(features.global_features.shape)  # (1, 1280)
        """
        # Handle different input types
        if isinstance(ref_image, ReferenceImage):
            # Check if features are cached
            if ref_image.features is not None:
                logger.debug("Using cached features")
                return ref_image.features

            image = ref_image.image
        else:
            image = ref_image

        # Use CLIP Vision encoder if available
        if self.clip_vision_encoder is not None:
            logger.info("Extracting image features with CLIP Vision")
            features = self.clip_vision_encoder.encode_image(
                image, return_patch_features=return_patch_features
            )
            return features
        else:
            # Fallback: placeholder features
            logger.warning("CLIP Vision not loaded, using placeholder features")
            hidden_dim = 1280  # CLIP-G dimension
            return ImageFeatures(
                global_features=np.zeros((1, hidden_dim)),
                patch_features=(
                    np.zeros((1, 256, hidden_dim)) if return_patch_features else None
                ),
            )

    def extract_features_batch(
        self,
        images: list[Union[Image.Image, np.ndarray]],
        return_patch_features: bool = False,
    ) -> list[ImageFeatures]:
        """
        Extract features from multiple images in batch.

        Args:
            images: List of images
            return_patch_features: Whether to return patch features

        Returns:
            List of ImageFeatures

        Example:
            >>> images = [Image.open(f"img{i}.png") for i in range(4)]
            >>> features_list = service.extract_features_batch(images)
        """
        if self.clip_vision_encoder is not None:
            return self.clip_vision_encoder.encode_images_batch(
                images, return_patch_features=return_patch_features
            )
        else:
            # Fallback: process one by one
            return [
                self.extract_features(img, return_patch_features)
                for img in images
            ]

    def apply_conditioning(
        self,
        ref_image: ReferenceImage,
        pipeline: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Apply IP-Adapter conditioning to pipeline.

        Args:
            ref_image: Reference image with features
            pipeline: Diffusers pipeline
            **kwargs: Additional parameters

        Returns:
            Modified pipeline or kwargs

        Note:
            This is still a placeholder. Real IP-Adapter conditioning requires:
            1. IP-Adapter model loaded
            2. Integration with pipeline's cross-attention
            3. Feature projection layers
        """
        logger.info(f"Applying IP-Adapter conditioning (scale: {ref_image.scale:.2f})")

        # Extract features if not cached
        if ref_image.features is None:
            ref_image.features = self.extract_features(ref_image)

        # TODO: Real implementation would:
        # 1. Project features with IP-Adapter projection layers
        # 2. Inject into pipeline's cross-attention
        # 3. Scale conditioning by ref_image.scale

        logger.warning("IP-Adapter conditioning is placeholder - real integration needed")
        return kwargs

    def is_clip_vision_loaded(self) -> bool:
        """Check if CLIP Vision encoder is loaded."""
        return self.clip_vision_encoder is not None

    def get_embedding_dim(self) -> int:
        """Get CLIP Vision embedding dimension."""
        if self.clip_vision_encoder is not None:
            return self.clip_vision_encoder.get_embedding_dim()
        return 1280  # Default CLIP-G

    def to(self, device: str) -> "IPAdapterService":
        """Move models to device."""
        if self.clip_vision_encoder is not None:
            self.clip_vision_encoder.to(device)
        return self
