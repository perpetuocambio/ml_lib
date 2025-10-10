"""IP-Adapter entities."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class IPAdapterVariant(Enum):
    """IP-Adapter model variant."""

    BASE = "base"
    """Standard IP-Adapter with 4 image tokens."""

    PLUS = "plus"
    """IP-Adapter Plus with 16 image tokens (higher fidelity)."""

    FACE_ID = "faceid"
    """Specialized for face preservation."""

    FULL_FACE = "full_face"
    """Full face structure preservation."""


@dataclass
class IPAdapterConfig:
    """Configuration for IP-Adapter."""

    model_id: str
    variant: IPAdapterVariant
    scale: float = 1.0
    """Scale of image conditioning (0.0 to 1.5)."""

    num_tokens: Optional[int] = None
    """Number of image tokens (auto-detected from variant if None)."""

    def __post_init__(self):
        """Validate and set defaults."""
        if not 0.0 <= self.scale <= 1.5:
            raise ValueError("scale must be between 0.0 and 1.5")

        # Auto-detect num_tokens from variant
        if self.num_tokens is None:
            if self.variant == IPAdapterVariant.BASE:
                self.num_tokens = 4
            elif self.variant == IPAdapterVariant.PLUS:
                self.num_tokens = 16
            else:
                self.num_tokens = 4


@dataclass
class ImageFeatures:
    """Extracted image features."""

    global_features: np.ndarray
    """Global image features (1, hidden_dim)."""

    patch_features: Optional[np.ndarray] = None
    """Patch-level features (num_patches, hidden_dim)."""

    cls_token: Optional[np.ndarray] = None
    """CLS token from vision transformer."""


@dataclass
class ReferenceImage:
    """Reference image for IP-Adapter."""

    image: np.ndarray
    """Image as numpy array (H, W, C)."""

    features: Optional[ImageFeatures] = None
    """Pre-extracted features (cached)."""

    scale: float = 1.0
    """Strength of this reference."""


__all__ = [
    "IPAdapterVariant",
    "IPAdapterConfig",
    "ImageFeatures",
    "ReferenceImage",
]
