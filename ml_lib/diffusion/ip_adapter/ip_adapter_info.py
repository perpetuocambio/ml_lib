"""Value objects for IP-Adapter handler."""

from dataclasses import dataclass
from typing import Optional, Protocol

from ml_lib.diffusion.models.ip_adapter.ip_adapter import IPAdapterConfig


@dataclass
class LoadedIPAdapterInfo:
    """Information about a loaded IP-Adapter model."""

    config: IPAdapterConfig
    model: Optional[object]  # Would be IPAdapterModel in production


class ModelRegistryProtocol(Protocol):
    """Protocol for model registry objects."""

    pass


class PipelineProtocol(Protocol):
    """Protocol for diffusion pipeline objects."""

    pass


class CLIPVisionEncoderProtocol(Protocol):
    """Protocol for CLIP Vision encoder objects."""

    def encode_image(
        self, image: object, return_patch_features: bool = True
    ) -> object: ...

    def encode_images_batch(
        self, images: list, return_patch_features: bool = False
    ) -> list: ...

    def get_embedding_dim(self) -> int: ...

    def to(self, device: str) -> "CLIPVisionEncoderProtocol": ...
