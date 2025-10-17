"""IP-Adapter service for image-based conditioning with real CLIP Vision."""

import logging
from pathlib import Path
from typing import Optional, Union
import numpy as np
from PIL import Image

from ml_lib.diffusion.domain.value_objects_models import (
    IPAdapterConfig,
    ReferenceImage,
    ImageFeatures,
)
from ml_lib.diffusion.domain.value_objects_models.value_objects import (
    LoadedIPAdapterInfo,
    ModelRegistryProtocol,
    PipelineProtocol,
    CLIPVisionEncoderProtocol,
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
        model_registry: Optional[ModelRegistryProtocol] = None,
        clip_vision_path: Optional[str] = None,
    ):
        """
        Initialize IP-Adapter service.

        Args:
            model_registry: Model registry for IP-Adapter models
            clip_vision_path: Path to CLIP Vision model (None = auto-detect)
        """
        self.model_registry = model_registry
        self._loaded_models: dict[str, LoadedIPAdapterInfo] = {}
        self.clip_vision_encoder: Optional[CLIPVisionEncoderProtocol] = None
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
            from ml_lib.diffusion.infrastructure.adapters.clip_vision_adapter import load_clip_vision

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
            Loads IP-Adapter weights from local path or HuggingFace.
            Model registry should provide paths to IP-Adapter weights.
        """
        logger.info(f"Loading IP-Adapter: {config.model_id} ({config.variant.value})")

        # Try to resolve path from model registry
        model_path = None
        if self.model_registry:
            try:
                # Try to get IP-Adapter model info from registry
                ip_adapter_info = self.model_registry.get_model_by_name(
                    config.model_id, "ip_adapter"
                )
                if ip_adapter_info and hasattr(ip_adapter_info, 'path'):
                    model_path = str(ip_adapter_info.path)
            except Exception as e:
                logger.debug(f"Could not resolve IP-Adapter from registry: {e}")

        # Store config and path
        self._loaded_models[config.model_id] = LoadedIPAdapterInfo(
            config=config,
            model=model_path  # Store path for later pipeline integration
        )

        if model_path:
            logger.info(f"✅ IP-Adapter loaded: {model_path}")
        else:
            logger.warning(
                f"IP-Adapter {config.model_id} registered but weights path not resolved. "
                "Pipeline integration will attempt to load from HuggingFace Hub."
            )

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
        pipeline: PipelineProtocol,
        ip_adapter_model_id: Optional[str] = None,
    ) -> PipelineProtocol:
        """
        Apply IP-Adapter conditioning to pipeline.

        Args:
            ref_image: Reference image with features
            pipeline: Diffusers pipeline
            ip_adapter_model_id: IP-Adapter model to use (None = use first loaded)

        Returns:
            Modified pipeline with IP-Adapter conditioning

        Implementation:
            1. Extracts image features using CLIP Vision
            2. Loads IP-Adapter weights into pipeline
            3. Sets up IP-Adapter image embeddings
            4. Configures scale for conditioning strength

        Example:
            >>> service = IPAdapterService()
            >>> service.load_clip_vision()
            >>> config = IPAdapterConfig(model_id="ip-adapter-plus", scale=0.8)
            >>> service.load_ip_adapter(config)
            >>> pipeline = service.apply_conditioning(ref_image, pipeline)
        """
        logger.info(f"Applying IP-Adapter conditioning (scale: {ref_image.scale:.2f})")

        # Extract features if not cached
        if ref_image.features is None:
            logger.debug("Extracting image features...")
            ref_image.features = self.extract_features(ref_image)

        # Get IP-Adapter model to use
        if not self._loaded_models:
            logger.warning("No IP-Adapter models loaded. Skipping conditioning.")
            return pipeline

        # Select model
        if ip_adapter_model_id and ip_adapter_model_id in self._loaded_models:
            model_info = self._loaded_models[ip_adapter_model_id]
        else:
            # Use first loaded model
            model_info = next(iter(self._loaded_models.values()))

        try:
            # Check if pipeline supports IP-Adapter
            if not hasattr(pipeline, 'load_ip_adapter'):
                logger.error(
                    "Pipeline does not support IP-Adapter (no load_ip_adapter method). "
                    "Ensure you're using a compatible diffusers pipeline."
                )
                return pipeline

            # Load IP-Adapter weights into pipeline
            if model_info.model:
                # Local path from registry
                logger.info(f"Loading IP-Adapter weights from: {model_info.model}")
                pipeline.load_ip_adapter(
                    model_info.model,
                    subfolder="",
                    weight_name=model_info.config.model_id
                )
            else:
                # Try HuggingFace Hub
                logger.info(f"Loading IP-Adapter from HuggingFace: {model_info.config.model_id}")
                pipeline.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="models",
                    weight_name=model_info.config.model_id
                )

            # Convert features to proper format for pipeline
            # IP-Adapter expects PIL image or numpy array
            if isinstance(ref_image.image, np.ndarray):
                from PIL import Image as PILImage
                # Convert numpy to PIL
                if ref_image.image.dtype != np.uint8:
                    img_uint8 = (ref_image.image * 255).astype(np.uint8)
                else:
                    img_uint8 = ref_image.image
                pil_image = PILImage.fromarray(img_uint8)
            else:
                pil_image = ref_image.image

            # Set IP-Adapter scale
            pipeline.set_ip_adapter_scale(ref_image.scale)

            # Store IP-Adapter image for pipeline to use during generation
            # The actual conditioning happens inside pipeline's forward pass
            if hasattr(pipeline, 'ip_adapter_image'):
                pipeline.ip_adapter_image = pil_image
            else:
                logger.warning(
                    "Pipeline doesn't have ip_adapter_image attribute. "
                    "IP-Adapter image will be passed during generation call."
                )

            logger.info(
                f"✅ IP-Adapter conditioning applied: "
                f"{model_info.config.model_id} (scale: {ref_image.scale:.2f})"
            )

        except AttributeError as e:
            logger.error(
                f"Pipeline missing IP-Adapter support: {e}. "
                "Install latest diffusers: pip install --upgrade diffusers"
            )
        except Exception as e:
            logger.error(f"Failed to apply IP-Adapter conditioning: {e}")
            logger.exception(e)

        return pipeline

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
