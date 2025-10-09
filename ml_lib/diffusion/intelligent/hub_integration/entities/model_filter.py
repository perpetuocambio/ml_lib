"""Model filter entities for search."""

from dataclasses import dataclass, field
from ml_lib.diffusion.intelligent.hub_integration.entities.model_metadata import (
    ModelType,
    BaseModel,
)
from enum import Enum


class SortBy(Enum):
    """Sort options for model search."""
    RELEVANCE = "relevance"
    POPULARITY = "popularity"
    DOWNLOADS = "downloads"
    RATING = "rating"
    NEWEST = "newest"
    UPDATED = "updated"


@dataclass
class ModelFilter:
    """Filter criteria for model search."""

    # Type filters
    model_type: ModelType | None = None
    base_model: BaseModel | None = None

    # Content filters
    tags: list[str] = field(default_factory=list)
    min_rating: float = 0.0
    min_downloads: int = 0

    # Technical filters
    max_size_gb: float | None = None
    formats: list[str] = field(default_factory=list)

    # HuggingFace specific
    task: str | None = None  # e.g., "text-to-image"
    library: str | None = None  # e.g., "diffusers"

    # Sorting
    sort_by: SortBy = SortBy.RELEVANCE

    def to_hf_params(self) -> dict:
        """Convert to HuggingFace API parameters."""
        params = {}

        if self.task:
            params["task"] = self.task
        if self.library:
            params["library"] = self.library

        # Tags become filter
        if self.tags:
            params["filter"] = ",".join(self.tags)

        # Sort
        if self.sort_by == SortBy.DOWNLOADS:
            params["sort"] = "downloads"
        elif self.sort_by == SortBy.UPDATED:
            params["sort"] = "lastModified"

        return params

    def to_civitai_params(self) -> dict:
        """Convert to CivitAI API parameters."""
        params = {}

        if self.model_type:
            # Map to CivitAI types
            type_map = {
                ModelType.BASE_MODEL: "Checkpoint",
                ModelType.LORA: "LORA",
                ModelType.EMBEDDING: "TextualInversion",
                ModelType.VAE: "VAE",
                ModelType.CONTROLNET: "Controlnet",
            }
            if self.model_type in type_map:
                params["types"] = type_map[self.model_type]

        if self.base_model and self.base_model != BaseModel.UNKNOWN:
            params["baseModels"] = self.base_model.value.upper()

        if self.tags:
            params["tag"] = ",".join(self.tags)

        # Sort
        if self.sort_by == SortBy.POPULARITY:
            params["sort"] = "Highest Rated"
        elif self.sort_by == SortBy.DOWNLOADS:
            params["sort"] = "Most Downloaded"
        elif self.sort_by == SortBy.NEWEST:
            params["sort"] = "Newest"

        return params

    def matches(self, metadata: "ModelMetadata") -> bool:
        """Check if metadata matches this filter."""
        # Type check
        if self.model_type and metadata.type != self.model_type:
            return False

        # Base model check
        if self.base_model and metadata.base_model != self.base_model:
            return False

        # Rating check
        if metadata.rating < self.min_rating:
            return False

        # Downloads check
        if metadata.download_count < self.min_downloads:
            return False

        # Size check
        if self.max_size_gb and metadata.size_gb > self.max_size_gb:
            return False

        # Tags check (any tag must match)
        if self.tags:
            if not any(tag in metadata.tags for tag in self.tags):
                return False

        # Format check
        if self.formats:
            if metadata.format.value not in self.formats:
                return False

        return True
