"""Model registry interface - Repository pattern."""

from typing import Protocol, runtime_checkable, Optional
from pathlib import Path


@runtime_checkable
class IModelRegistry(Protocol):
    """
    Protocol for model registry operations.

    Provides access to model metadata and files.
    """

    def get_lora_by_name(self, name: str) -> Optional[any]:
        """
        Get LoRA information by name.

        Args:
            name: LoRA model name

        Returns:
            LoRA info or None if not found
        """
        ...

    def get_loras_by_base_model(self, base_model: str) -> list[any]:
        """
        Get all LoRAs compatible with base model.

        Args:
            base_model: Base model architecture

        Returns:
            List of compatible LoRAs
        """
        ...

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """
        Get filesystem path for model.

        Args:
            model_name: Model name

        Returns:
            Path to model file or None if not found
        """
        ...
