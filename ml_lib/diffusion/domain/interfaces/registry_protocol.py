"""Protocol for model registry."""

from typing import Protocol, Optional, runtime_checkable

from ml_lib.diffusion.models.registry import ModelMetadata


@runtime_checkable
class ModelRegistryProtocol(Protocol):
    """Protocol for model registry implementations."""

    def search_models(
        self,
        query: str,
        model_type: Optional[str] = None,
        limit: int = 10
    ) -> list[ModelMetadata]:
        """
        Search for models matching query.

        Args:
            query: Search query
            model_type: Optional model type filter
            limit: Maximum results

        Returns:
            List of model metadata
        """
        ...

    def get_model_metadata(self, model_id: str) -> ModelMetadata:
        """
        Get metadata for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Model metadata object
        """
        ...

    def download_model(self, model_id: str, cache_dir: Optional[str] = None) -> str:
        """
        Download a model.

        Args:
            model_id: Model to download
            cache_dir: Cache directory

        Returns:
            Path to downloaded model
        """
        ...
