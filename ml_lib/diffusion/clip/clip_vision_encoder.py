from PIL import Image

import logging
from pathlib import Path
from typing import Optional, Union, Any

import numpy as np
import torch
import torchvision.transforms as T

from safetensors.torch import load_file
from transformers import CLIPVisionModel, CLIPImageProcessor

from ml_lib.diffusion.models import ImageFeatures

logger = logging.getLogger(__name__)


class CLIPVisionEncoder:
    """
    CLIP Vision encoder for extracting image features.

    Compatible with ComfyUI models:
    - clip_vision_g.safetensors (CLIP-G, 3.5GB)
    - clip_vision_h.safetensors (CLIP-H, 1.2GB)

    Example:
        >>> encoder = CLIPVisionEncoder.from_pretrained(
        ...     "/src/ComfyUI/models/clip_vision/clip_vision_g.safetensors"
        ... )
        >>> image = Image.open("reference.png")
        >>> features = encoder.encode_image(image)
        >>> print(features.global_features.shape)  # (1, 1280)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Optional[Any] = None,
        device: str = "cuda",
    ):
        """
        Initialize CLIP Vision encoder.

        Args:
            model: CLIP Vision model
            processor: Image processor/transform
            device: Device to run on
        """
        self.model = model.to(device).eval()
        self.processor = processor
        self.device = device
        self.hidden_size = self._detect_hidden_size()

        logger.info(
            f"CLIPVisionEncoder initialized (hidden_size={self.hidden_size}, device={device})"
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ) -> "CLIPVisionEncoder":
        """
        Load CLIP Vision model from file.

        Args:
            model_path: Path to .safetensors file
            device: Device to load on
            torch_dtype: Data type (fp16 recommended)

        Returns:
            CLIPVisionEncoder instance

        Example:
            >>> encoder = CLIPVisionEncoder.from_pretrained(
            ...     "/src/ComfyUI/models/clip_vision/clip_vision_g.safetensors",
            ...     device="cuda",
            ...     torch_dtype=torch.float16
            ... )
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading CLIP Vision from {model_path}...")

        try:
            # Load state dict
            state_dict = load_file(str(model_path))

            # Detect model type from file size or name
            file_size_gb = model_path.stat().st_size / (1024**3)
            model_name = model_path.stem

            if "clip_vision_g" in model_name or file_size_gb > 2.0:
                # CLIP-G (large model)
                logger.info("Detected CLIP-G model")
                model_config = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
            elif "clip_vision_h" in model_name or file_size_gb > 0.5:
                # CLIP-H
                logger.info("Detected CLIP-H model")
                model_config = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
            else:
                # Default to CLIP-H
                logger.warning("Unknown model type, defaulting to CLIP-H")
                model_config = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

            # Load model architecture
            model = CLIPVisionModel.from_pretrained(
                model_config,
                torch_dtype=torch_dtype,
            )

            # Load weights from safetensors
            model.load_state_dict(state_dict, strict=False)

            # Load processor
            processor = CLIPImageProcessor.from_pretrained(model_config)

            logger.info(f"✅ CLIP Vision loaded successfully from {model_path.name}")

            return cls(model=model, processor=processor, device=device)

        except Exception as e:
            logger.error(f"Failed to load CLIP Vision: {e}")
            raise

    def _detect_hidden_size(self) -> int:
        """Detect hidden size from model."""
        try:
            return self.model.config.hidden_size
        except AttributeError:
            # Fallback
            return 1280  # CLIP-G default

    def preprocess_image(
        self, image: Union[Image.Image, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess image for CLIP Vision.

        Args:
            image: PIL Image, numpy array, or torch tensor

        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, torch.Tensor):
            # Assume CHW format, convert to PIL
            if image.dim() == 3:
                image = image.permute(1, 2, 0).cpu().numpy()
                image = Image.fromarray((image * 255).astype(np.uint8))

        if self.processor is not None:
            # Use official processor
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
        else:
            # Fallback: manual preprocessing
            transform = T.Compose(
                [
                    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )
            pixel_values = transform(image).unsqueeze(0).to(self.device)

        return pixel_values

    @torch.no_grad()
    def encode_image(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        return_patch_features: bool = True,
    ) -> ImageFeatures:
        """
        Encode image to features.

        Args:
            image: Input image
            return_patch_features: Whether to return patch-level features

        Returns:
            ImageFeatures with global and optionally patch features

        Example:
            >>> image = Image.open("reference.png")
            >>> features = encoder.encode_image(image)
            >>> print(f"Global: {features.global_features.shape}")
            >>> print(f"Patches: {features.patch_features.shape}")
        """
        # Preprocess
        pixel_values = self.preprocess_image(image)

        # Encode
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)

        # Extract features
        # Global features: CLS token from last hidden state
        last_hidden_state = outputs.last_hidden_state  # (1, num_patches+1, hidden)
        cls_token = last_hidden_state[:, 0, :]  # (1, hidden)

        # Patch features: all tokens except CLS
        patch_features = None
        if return_patch_features:
            patch_features = last_hidden_state[:, 1:, :]  # (1, num_patches, hidden)
            patch_features = patch_features.cpu().numpy()

        # Convert to numpy
        global_features = cls_token.cpu().numpy()

        logger.debug(
            f"Encoded image: global={global_features.shape}, "
            f"patches={patch_features.shape if patch_features is not None else 'None'}"
        )

        return ImageFeatures(
            global_features=global_features,
            patch_features=patch_features,
            cls_token=global_features,  # Same as global for CLIP
        )

    @torch.no_grad()
    def encode_images_batch(
        self,
        images: list[Union[Image.Image, np.ndarray]],
        return_patch_features: bool = False,
    ) -> list[ImageFeatures]:
        """
        Encode multiple images in batch.

        Args:
            images: List of images
            return_patch_features: Whether to return patch features

        Returns:
            List of ImageFeatures

        Example:
            >>> images = [Image.open(f"img{i}.png") for i in range(4)]
            >>> features_list = encoder.encode_images_batch(images)
        """
        if not images:
            return []

        # Preprocess all images
        pixel_values = torch.cat([self.preprocess_image(img) for img in images], dim=0)

        # Encode batch
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)

        # Extract features for each image
        last_hidden_state = outputs.last_hidden_state  # (B, num_patches+1, hidden)
        batch_size = last_hidden_state.shape[0]

        features_list = []
        for i in range(batch_size):
            cls_token = last_hidden_state[i : i + 1, 0, :]  # (1, hidden)

            patch_features = None
            if return_patch_features:
                patch_features = last_hidden_state[i : i + 1, 1:, :]
                patch_features = patch_features.cpu().numpy()

            global_features = cls_token.cpu().numpy()

            features_list.append(
                ImageFeatures(
                    global_features=global_features,
                    patch_features=patch_features,
                    cls_token=global_features,
                )
            )

        return features_list

    def get_embedding_dim(self) -> int:
        """Get dimension of embeddings."""
        return self.hidden_size

    def to(self, device: str) -> "CLIPVisionEncoder":
        """Move model to device."""
        self.model = self.model.to(device)
        self.device = device
        logger.info(f"Moved CLIPVisionEncoder to {device}")
        return self
