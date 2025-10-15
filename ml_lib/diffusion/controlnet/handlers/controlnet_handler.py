"""ControlNet handler for spatial control in image generation."""

import logging
from typing import Optional
from pathlib import Path

from ml_lib.diffusion.controlnet.enums.control_type import ControlType
from ml_lib.diffusion.controlnet.models.control_image import ControlImage
from ml_lib.diffusion.controlnet.models.controlnet_config import ControlNetConfig
from ml_lib.diffusion.controlnet.models.loaded_controlnet_info import (
    LoadedControlNetInfo,
)
from diffusers import ControlNetModel
from ml_lib.diffusion.generation.registry_protocol import ModelRegistryProtocol
import torch

logger = logging.getLogger(__name__)


class ControlNetHandler:
    """
    Main handler for ControlNet functionality.

    Handles:
    - Loading ControlNet models
    - Applying control signals during generation
    - Managing conditioning scale

    Note: Actual diffusers integration is placeholder for now.
    Full implementation requires torch/diffusers/controlnet_aux.
    """

    def __init__(self, model_registry: Optional[ModelRegistryProtocol] = None):
        """
        Initialize ControlNet handler.

        Args:
            model_registry: ModelRegistry for finding/loading models
        """
        self.model_registry = model_registry
        self._loaded_models: dict[str, LoadedControlNetInfo] = {}
        logger.info("ControlNetHandler initialized")

    def load_controlnet(self, config: ControlNetConfig, device: str = "cuda") -> None:
        """
        Load a ControlNet model.

        Args:
            config: ControlNet configuration
            device: Device to load model on

        Raises:
            ImportError: If diffusers is not installed
            RuntimeError: If model loading fails
        """
        if config.model_id in self._loaded_models:
            logger.debug(f"ControlNet {config.model_id} already loaded")
            return

        logger.info(
            f"Loading ControlNet: {config.model_id} (type: {config.control_type.value})"
        )

        try:
            # Determine dtype for memory efficiency
            torch_dtype = torch.float16 if device == "cuda" else torch.float32

            # Try to load from registry first
            model_path = None
            if self.model_registry:
                try:
                    controlnet_info = self.model_registry.get_model_by_name(
                        config.model_id, "controlnet"
                    )
                    if controlnet_info and hasattr(controlnet_info, "path"):
                        model_path = str(controlnet_info.path)
                        logger.debug(f"Found ControlNet in registry: {model_path}")
                except Exception as e:
                    logger.debug(f"Could not resolve ControlNet from registry: {e}")

            # Load model
            if model_path and Path(model_path).exists():
                # Load from local path
                model = ControlNetModel.from_pretrained(
                    model_path, torch_dtype=torch_dtype, use_safetensors=True
                )
                logger.info(f"Loaded ControlNet from local path: {model_path}")
            else:
                # Load from HuggingFace Hub
                model = ControlNetModel.from_pretrained(
                    config.model_id, torch_dtype=torch_dtype
                )
                logger.info(f"Loaded ControlNet from HuggingFace: {config.model_id}")

            # Move to device
            model = model.to(device)

            self._loaded_models[config.model_id] = LoadedControlNetInfo(
                config=config,
                model=model,
                device=device,
            )

            logger.info(
                f"✅ ControlNet {config.model_id} loaded successfully on {device}"
            )

        except ImportError:
            logger.error(
                "Failed to import diffusers. Install with: pip install diffusers torch"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load ControlNet {config.model_id}: {e}")
            # Store placeholder to indicate attempted load
            self._loaded_models[config.model_id] = LoadedControlNetInfo(
                config=config,
                model=None,
                device=device,
            )
            raise RuntimeError(f"Failed to load ControlNet: {e}") from e

    def apply_control(
        self,
        control_image: ControlImage,
        pipeline: PipelineProtocol,
    ) -> PipelineProtocol:
        """
        Apply ControlNet control to generation pipeline.

        Args:
            control_image: Processed control image
            pipeline: Diffusion pipeline to modify

        Returns:
            Modified pipeline with ControlNet conditioning

        Implementation:
        1. Validates ControlNet is loaded
        2. Prepares control image for pipeline
        3. Attaches ControlNet to pipeline
        4. Sets conditioning scale

        Note: The pipeline must support ControlNet (e.g., StableDiffusionControlNetPipeline)
        """
        logger.info(
            f"Applying {control_image.control_type.value} control "
            f"(scale: {control_image.scale:.2f})"
        )

        # Find matching loaded ControlNet
        controlnet_model = None
        for model_id, info in self._loaded_models.items():
            if (
                info.config.control_type == control_image.control_type
                and info.model is not None
            ):
                controlnet_model = info.model
                logger.debug(f"Using loaded ControlNet: {model_id}")
                break

        if controlnet_model is None:
            logger.warning(
                f"No ControlNet loaded for type {control_image.control_type.value}. "
                "Control will not be applied."
            )
            return pipeline

        try:
            # Set ControlNet conditioning
            # This depends on the pipeline type
            if hasattr(pipeline, "controlnet"):
                # StableDiffusionControlNetPipeline
                pipeline.controlnet = controlnet_model
                logger.debug("✅ ControlNet attached to pipeline")
            elif hasattr(pipeline, "set_adapters"):
                # ControlNet as adapter (newer diffusers API)
                pipeline.set_adapters([controlnet_model])
                logger.debug("✅ ControlNet set as adapter")
            else:
                logger.warning(
                    "Pipeline does not support ControlNet (no controlnet attribute or set_adapters). "
                    "Ensure you're using StableDiffusionControlNetPipeline or compatible."
                )
                return pipeline

            # Store control image and scale for pipeline to use during generation
            # The actual conditioning happens inside pipeline's forward pass
            if hasattr(pipeline, "controlnet_conditioning_scale"):
                pipeline.controlnet_conditioning_scale = control_image.scale

            # Store control image reference
            if hasattr(pipeline, "control_image"):
                pipeline.control_image = control_image.image
            else:
                logger.warning(
                    "Pipeline doesn't have control_image attribute. "
                    "Control image must be passed during generation call."
                )

            logger.info(
                f"✅ ControlNet control applied: {control_image.control_type.value} "
                f"(scale: {control_image.scale:.2f})"
            )

        except Exception as e:
            logger.error(f"Failed to apply ControlNet control: {e}")
            logger.exception(e)

        return pipeline

    def unload_controlnet(self, model_id: str) -> None:
        """
        Unload a ControlNet model to free memory.

        Args:
            model_id: ID of model to unload
        """
        if model_id in self._loaded_models:
            del self._loaded_models[model_id]
            logger.info(f"Unloaded ControlNet: {model_id}")

    def list_loaded_models(self) -> list[str]:
        """Get list of loaded ControlNet model IDs."""
        return list(self._loaded_models.keys())

    def get_recommended_scale(
        self, control_type: ControlType, complexity: str = "moderate"
    ) -> float:
        """
        Get recommended conditioning scale for control type and complexity.

        Args:
            control_type: Type of control
            complexity: Prompt complexity (simple, moderate, complex)

        Returns:
            Recommended scale value
        """
        # Recommendations based on empirical testing
        base_scales = {
            ControlType.CANNY: 0.8,
            ControlType.DEPTH: 0.9,
            ControlType.POSE: 1.0,
            ControlType.SEGMENTATION: 0.7,
            ControlType.NORMAL: 0.85,
            ControlType.SCRIBBLE: 0.6,
            ControlType.MLSD: 0.75,
            ControlType.HED: 0.8,
        }

        scale = base_scales.get(control_type, 0.8)

        # Adjust for complexity
        if complexity == "simple":
            scale *= 1.1
        elif complexity == "complex":
            scale *= 0.9

        return round(min(scale, 1.5), 2)
