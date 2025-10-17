"""In-Memory Model Repository - For testing and development.

This implementation stores LoRAs in memory, making it perfect for:
- Unit tests (no DB required)
- Integration tests
- Development/prototyping
- Demo applications
"""

from typing import Optional
from ml_lib.diffusion.domain.repositories.model_repository import IModelRepository
from ml_lib.diffusion.domain.entities.lora import LoRA


class InMemoryModelRepository(IModelRepository):
    """
    In-memory implementation of IModelRepository.

    Perfect for testing - no database required.
    """

    def __init__(self):
        """Initialize empty repository."""
        self._loras: dict[str, LoRA] = {}

    def get_lora_by_name(self, name: str) -> Optional[LoRA]:
        """Get LoRA by name."""
        return self._loras.get(name)

    def get_all_loras(self) -> list[LoRA]:
        """Get all LoRAs."""
        return list(self._loras.values())

    def get_loras_by_base_model(self, base_model: str) -> list[LoRA]:
        """Get LoRAs compatible with base model."""
        return [
            lora for lora in self._loras.values()
            if lora.is_compatible_with(base_model)
        ]

    def get_loras_by_tags(self, tags: list[str]) -> list[LoRA]:
        """Get LoRAs that have any of the tags."""
        tags_lower = [t.lower() for t in tags]
        matching = []

        for lora in self._loras.values():
            lora_tags_lower = [t.lower() for t in lora.tags]
            if any(tag in lora_tags_lower for tag in tags_lower):
                matching.append(lora)

        return matching

    def get_popular_loras(self, limit: int = 10) -> list[LoRA]:
        """Get most popular LoRAs."""
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
        """Search LoRAs by query."""
        loras = self.get_all_loras()

        # Filter by base model
        if base_model:
            loras = [l for l in loras if l.is_compatible_with(base_model)]

        # Filter by rating
        loras = [l for l in loras if l.rating >= min_rating]

        # Filter by query
        query_lower = query.lower()
        matching = []

        for lora in loras:
            # Search in name
            if query_lower in lora.name.lower():
                matching.append(lora)
                continue

            # Search in tags
            if any(query_lower in tag.lower() for tag in lora.tags):
                matching.append(lora)
                continue

            # Search in trigger words
            if any(query_lower in tw.lower() for tw in lora.trigger_words):
                matching.append(lora)
                continue

        return matching[:limit]

    def add_lora(self, lora: LoRA) -> None:
        """Add a new LoRA."""
        if lora.name in self._loras:
            raise ValueError(f"LoRA '{lora.name}' already exists")

        self._loras[lora.name] = lora

    def update_lora(self, lora: LoRA) -> None:
        """Update existing LoRA."""
        if lora.name not in self._loras:
            raise ValueError(f"LoRA '{lora.name}' not found")

        self._loras[lora.name] = lora

    def delete_lora(self, name: str) -> bool:
        """Delete LoRA by name."""
        if name in self._loras:
            del self._loras[name]
            return True
        return False

    def count_loras(self) -> int:
        """Get total count."""
        return len(self._loras)

    def clear(self) -> None:
        """Clear all LoRAs (useful for testing)."""
        self._loras.clear()

    def seed_with_samples(self) -> None:
        """
        Seed repository with sample LoRAs for testing/demo.

        Creates fake LoRAs with realistic data.
        """
        from pathlib import Path
        import tempfile

        # Create temp directory for fake files
        temp_dir = Path(tempfile.gettempdir()) / "fake_loras"
        temp_dir.mkdir(exist_ok=True)

        # Sample LoRAs
        samples = [
            {
                "name": "anime_style_v2",
                "base_model": "SDXL",
                "trigger_words": ["anime", "manga style"],
                "tags": ["anime", "illustration", "2d", "cartoon"],
                "download_count": 50000,
                "rating": 4.7,
                "weight": 0.8,
            },
            {
                "name": "realistic_details",
                "base_model": "SDXL",
                "trigger_words": ["photorealistic", "detailed"],
                "tags": ["realistic", "photo", "detailed"],
                "download_count": 30000,
                "rating": 4.5,
                "weight": 1.0,
            },
            {
                "name": "fantasy_art",
                "base_model": "SDXL",
                "trigger_words": ["fantasy", "magical"],
                "tags": ["fantasy", "magic", "illustration"],
                "download_count": 20000,
                "rating": 4.3,
                "weight": 0.9,
            },
            {
                "name": "cyberpunk_style",
                "base_model": "SDXL",
                "trigger_words": ["cyberpunk", "neon"],
                "tags": ["cyberpunk", "sci-fi", "neon", "futuristic"],
                "download_count": 15000,
                "rating": 4.2,
                "weight": 0.85,
            },
            {
                "name": "oil_painting",
                "base_model": "SDXL",
                "trigger_words": ["oil painting", "classical art"],
                "tags": ["painting", "classical", "artistic"],
                "download_count": 10000,
                "rating": 4.6,
                "weight": 0.75,
            },
        ]

        for sample in samples:
            # Create fake file
            fake_file = temp_dir / f"{sample['name']}.safetensors"
            if not fake_file.exists():
                fake_file.write_text("fake lora data")

            # Create LoRA
            lora = LoRA.create(
                name=sample["name"],
                path=fake_file,
                base_model=sample["base_model"],
                weight=sample["weight"],
                trigger_words=sample["trigger_words"],
                tags=sample["tags"],
                download_count=sample["download_count"],
                rating=sample["rating"],
            )

            self.add_lora(lora)
