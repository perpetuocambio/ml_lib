"""
CLIP Vision Encoder - Real implementation for IP-Adapter.

Supports CLIP-G and CLIP-H models from ComfyUI.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Any

import numpy as np
import torch
import torchvision.transforms as T

from safetensors.torch import load_file
from transformers import CLIPVisionModel, CLIPImageProcessor

from ml_lib.diffusion.models import ImageFeatures
from ml_lib.diffusion.config import detect_comfyui_installation

logger = logging.getLogger(__name__)


# Convenience function for quick loading
def load_clip_vision(
    model_path: Optional[str] = None,
    device: str = "cuda",
    auto_detect_comfyui: bool = True,
    search_paths: Optional[list[Path | str]] = None,
) -> CLIPVisionEncoder:
    """
    Convenience function to load CLIP Vision encoder.

    Args:
        model_path: Path to model (None = auto-detect)
        device: Device to use
        auto_detect_comfyui: Auto-detect from ComfyUI
        search_paths: Custom search paths for ComfyUI

    Returns:
        CLIPVisionEncoder instance

    Example:
        >>> # Auto-detect from ComfyUI
        >>> encoder = load_clip_vision()

        >>> # Explicit path
        >>> encoder = load_clip_vision("/path/to/clip_vision_g.safetensors")

        >>> # Custom search
        >>> encoder = load_clip_vision(
        ...     search_paths=["/opt/comfyui", "/home/user/comfyui"]
        ... )
    """
    if model_path is None and auto_detect_comfyui:
        # Auto-detect from ComfyUI
        comfyui_root = detect_comfyui_installation(search_paths)
        if comfyui_root:
            # Prefer CLIP-G for SDXL
            clip_g = comfyui_root / "models/clip_vision/clip_vision_g.safetensors"
            clip_h = comfyui_root / "models/clip_vision/clip_vision_h.safetensors"

            if clip_g.exists():
                model_path = str(clip_g)
                logger.info(f"Auto-detected CLIP-G at {model_path}")
            elif clip_h.exists():
                model_path = str(clip_h)
                logger.info(f"Auto-detected CLIP-H at {model_path}")
            else:
                raise FileNotFoundError(
                    f"No CLIP Vision models found in {comfyui_root}/models/clip_vision"
                )
        else:
            raise ValueError(
                "ComfyUI not found. Provide model_path explicitly or "
                "add ComfyUI location to search_paths."
            )

    if model_path is None:
        raise ValueError("model_path is required when auto_detect_comfyui=False")

    return CLIPVisionEncoder.from_pretrained(model_path, device=device)
