"""Adapter that makes ModelRegistry implement IModelRepository.

This adapter bridges the existing ModelRegistry with the new Repository interface,
allowing us to use the new clean architecture without rewriting everything at once.
"""

from typing import Optional
from pathlib import Path
from ml_lib.diffusion.domain.repositories.model_repository import IModelRepository
from ml_lib.diffusion.domain.entities.lora import LoRA


class ModelRegistryAdapter(IModelRepository):
    """
    Adapter that wraps ModelRegistry to implement IModelRepository.

    This allows gradual migration:
    - New code depends on IModelRepository (interface)
    - This adapter delegates to existing ModelRegistry
    - Later we can swap for SQLiteModelRepository without changing domain code
    """

    def __init__(self, model_registry):
        """
        Initialize adapter.

        Args:
            model_registry: Existing ModelRegistry instance
        """
        self._registry = model_registry

    def get_lora_by_name(self, name: str) -> Optional[LoRA]:
        """Get LoRA by name."""
        # Delegate to existing registry
        lora_info = self._registry.get_lora_by_name(name)

        if lora_info is None:
            return None

        # Convert old LoRAInfo to new LoRA entity
        return self._convert_to_lora(lora_info)

    def get_all_loras(self) -> list[LoRA]:
        """Get all LoRAs."""
        # Get from registry (method may not exist yet)
        try:
            lora_infos = self._registry.get_all_loras()
            return [self._convert_to_lora(info) for info in lora_infos]
        except AttributeError:
            # Fallback: return empty if method doesn't exist
            return []

    def get_loras_by_base_model(self, base_model: str) -> list[LoRA]:
        """Get LoRAs by base model."""
        try:
            lora_infos = self._registry.get_loras_by_base_model(base_model)
            return [self._convert_to_lora(info) for info in lora_infos]
        except AttributeError:
            # Fallback
            return []

    def get_loras_by_tags(self, tags: list[str]) -> list[LoRA]:
        """Get LoRAs by tags."""
        # Not implemented in old registry, return empty
        return []

    def get_popular_loras(self, limit: int = 10) -> list[LoRA]:
        """Get popular LoRAs."""
        # Get all and sort by popularity
        all_loras = self.get_all_loras()
        sorted_loras = sorted(
            all_loras,
            key=lambda l: l.get_popularity_score(),
            reverse=True
        )
        return sorted_loras[:limit]

    def search_loras(
        self,
        query: str,
        base_model: Optional[str] = None,
        min_rating: float = 0.0,
        limit: int = 20,
    ) -> list[LoRA]:
        """Search LoRAs."""
        # Simple implementation: filter all LoRAs
        loras = self.get_all_loras()

        # Filter by base model
        if base_model:
            loras = [l for l in loras if l.is_compatible_with(base_model)]

        # Filter by rating
        loras = [l for l in loras if l.rating >= min_rating]

        # Filter by query (search in name, tags, triggers)
        query_lower = query.lower()
        matching = []
        for lora in loras:
            if (query_lower in lora.name.lower() or
                any(query_lower in tag.lower() for tag in lora.tags) or
                any(query_lower in tw.lower() for tw in lora.trigger_words)):
                matching.append(lora)

        return matching[:limit]

    def add_lora(self, lora: LoRA) -> None:
        """Add LoRA - not implemented in adapter."""
        raise NotImplementedError(
            "ModelRegistryAdapter is read-only. "
            "Use SQLiteModelRepository for write operations."
        )

    def update_lora(self, lora: LoRA) -> None:
        """Update LoRA - not implemented in adapter."""
        raise NotImplementedError(
            "ModelRegistryAdapter is read-only. "
            "Use SQLiteModelRepository for write operations."
        )

    def delete_lora(self, name: str) -> bool:
        """Delete LoRA - not implemented in adapter."""
        raise NotImplementedError(
            "ModelRegistryAdapter is read-only. "
            "Use SQLiteModelRepository for write operations."
        )

    def count_loras(self) -> int:
        """Count LoRAs."""
        return len(self.get_all_loras())

    def _convert_to_lora(self, lora_info) -> LoRA:
        """
        Convert old LoRAInfo to new LoRA entity.

        Args:
            lora_info: Old LoRAInfo from ModelRegistry

        Returns:
            New LoRA entity
        """
        # Extract fields from old LoRAInfo
        # (Field names may vary, adjust as needed)
        name = getattr(lora_info, 'name', 'unknown')
        path = getattr(lora_info, 'path', Path('unknown'))
        base_model = getattr(lora_info, 'base_model', 'SDXL')
        alpha = getattr(lora_info, 'alpha', 1.0)
        trigger_words = getattr(lora_info, 'trigger_words', [])
        tags = getattr(lora_info, 'tags', [])
        download_count = getattr(lora_info, 'download_count', 0)
        rating = getattr(lora_info, 'rating', 0.0)

        # Create new LoRA entity
        return LoRA.create(
            name=name,
            path=Path(path) if path else Path('unknown'),
            base_model=base_model,
            weight=alpha,
            trigger_words=trigger_words or [],
            tags=tags or [],
            download_count=download_count,
            rating=rating,
        )
