from enum import Enum


class CLIPVisionModelType(Enum):
    """CLIP Vision model variants."""

    CLIP_G = "clip_vision_g"  # 3.5GB - For SDXL IP-Adapter
    CLIP_H = "clip_vision_h"  # 1.2GB - For SD 1.5 IP-Adapter
