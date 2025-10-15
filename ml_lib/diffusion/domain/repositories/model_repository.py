"""Model Repository Interface - Repository Pattern.

This interface defines the contract for accessing models (LoRAs, checkpoints, etc.)
from persistent storage.

Following Repository pattern:
- Abstracts data access
- Allows swapping implementations (SQLite, MongoDB, API, etc.)
- Domain depends on interface, not implementation
"""

from typing import Protocol, Optional, runtime_checkable
from pathlib import Path
from ml_lib.diffusion.domain.entities.lora import LoRA


@runtime_checkable
class IModelRepository(Protocol):
    """
    Repository for accessing model metadata and files.

    This is the main interface that domain services depend on.
    Implementations handle actual data access (SQLite, API, etc.)
    """

    def get_lora_by_name(self, name: str) -> Optional[LoRA]:
        """
        Get LoRA by name.

        Args:
            name: LoRA name

        Returns:
            LoRA entity or None if not found
        """
        ...

    def get_all_loras(self) -> list[LoRA]:
        """
        Get all available LoRAs.

        Returns:
            List of all LoRA entities
        """
        ...

    def get_loras_by_base_model(self, base_model: str) -> list[LoRA]:
        """
        Get LoRAs compatible with base model.

        Args:
            base_model: Base model architecture (e.g., "SDXL", "SD15")

        Returns:
            List of compatible LoRA entities
        """
        ...

    def get_loras_by_tags(self, tags: list[str]) -> list[LoRA]:
        """
        Get LoRAs that have any of the specified tags.

        Args:
            tags: List of tags to search for

        Returns:
            List of LoRAs matching tags
        """
        ...

    def get_popular_loras(self, limit: int = 10) -> list[LoRA]:
        """
        Get most popular LoRAs by download count.

        Args:
            limit: Maximum number to return

        Returns:
            List of popular LoRAs, sorted by popularity
        """
        ...

    def search_loras(
        self,
        query: str,
        base_model: Optional[str] = None,
        min_rating: float = 0.0,
        limit: int = 20,
    ) -> list[LoRA]:
        """
        Search LoRAs by query.

        Args:
            query: Search query (searches name, tags, trigger words)
            base_model: Optional filter by base model
            min_rating: Minimum rating filter
            limit: Maximum results

        Returns:
            List of matching LoRAs
        """
        ...

    def add_lora(self, lora: LoRA) -> None:
        """
        Add a new LoRA to repository.

        Args:
            lora: LoRA entity to add
        """
        ...

    def update_lora(self, lora: LoRA) -> None:
        """
        Update existing LoRA.

        Args:
            lora: LoRA entity with updated data
        """
        ...

    def delete_lora(self, name: str) -> bool:
        """
        Delete LoRA by name.

        Args:
            name: LoRA name

        Returns:
            True if deleted, False if not found
        """
        ...

    def count_loras(self) -> int:
        """
        Get total count of LoRAs.

        Returns:
            Number of LoRAs in repository
        """
        ...


class ILoRARepository(Protocol):
    """
    Specialized repository interface for LoRAs.

    Same as IModelRepository for now, but allows future LoRA-specific extensions.
    """

    def get_lora_by_name(self, name: str) -> Optional[LoRA]:
        """Get LoRA by name."""
        ...

    def get_all_loras(self) -> list[LoRA]:
        """Get all LoRAs."""
        ...

    def get_loras_by_base_model(self, base_model: str) -> list[LoRA]:
        """Get LoRAs by base model."""
        ...

    def get_loras_by_tags(self, tags: list[str]) -> list[LoRA]:
        """Get LoRAs by tags."""
        ...

    def get_popular_loras(self, limit: int = 10) -> list[LoRA]:
        """Get popular LoRAs."""
        ...

    def search_loras(
        self,
        query: str,
        base_model: Optional[str] = None,
        min_rating: float = 0.0,
        limit: int = 20,
    ) -> list[LoRA]:
        """Search LoRAs."""
        ...

    def add_lora(self, lora: LoRA) -> None:
        """Add LoRA."""
        ...

    def update_lora(self, lora: LoRA) -> None:
        """Update LoRA."""
        ...

    def delete_lora(self, name: str) -> bool:
        """Delete LoRA."""
        ...

    def count_loras(self) -> int:
        """Count LoRAs."""
        ...
