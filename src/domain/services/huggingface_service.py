"""HuggingFace Hub integration service."""

import hashlib
import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download, scan_cache_dir
from huggingface_hub.utils import HfHubHTTPError

from ml_lib.diffusion.domain.value_objects_models import (
    ModelMetadata,
    ModelFilter,
    DownloadResult,
    DownloadStatus,
    Source,
    ModelType,
    BaseModel,
    ModelFormat,
)

logger = logging.getLogger(__name__)


class HuggingFaceHubService:
    """Service for interacting with HuggingFace Hub."""

    def __init__(
        self,
        token: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize HuggingFace Hub service.

        Args:
            token: HuggingFace API token (optional, for private models)
            cache_dir: Custom cache directory (default: ~/.cache/huggingface)
        """
        self.client = HfApi(token=token)
        self.token = token
        self.cache_dir = cache_dir or Path.home() / ".cache" / "huggingface"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized HuggingFace Hub service with cache: {self.cache_dir}")

    def search_models(
        self,
        query: str,
        filter: Optional[ModelFilter] = None,
        limit: int = 20,
    ) -> list[ModelMetadata]:
        """
        Search models on HuggingFace Hub.

        Args:
            query: Search query string
            filter: Optional filter criteria
            limit: Maximum number of results

        Returns:
            List of model metadata
        """
        try:
            # Build search parameters
            params = {"search": query, "limit": limit}

            if filter:
                params.update(filter.to_hf_params())

            # Search models
            models = self.client.list_models(**params)

            results = []
            for model in models:
                try:
                    metadata = self._model_to_metadata(model)

                    # Apply additional filters
                    if filter and not filter.matches(metadata):
                        continue

                    results.append(metadata)

                    if len(results) >= limit:
                        break

                except Exception as e:
                    logger.warning(f"Failed to process model {model.id}: {e}")
                    continue

            logger.info(f"Found {len(results)} models for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Failed to search models: {e}")
            return []

    def download_model(
        self,
        model_id: str,
        revision: str = "main",
        allow_patterns: Optional[list[str]] = None,
        ignore_patterns: Optional[list[str]] = None,
    ) -> DownloadResult:
        """
        Download model from HuggingFace Hub.

        Args:
            model_id: HuggingFace model ID (e.g., "stabilityai/sdxl-base-1.0")
            revision: Git revision (branch, tag, or commit)
            allow_patterns: Patterns to include (e.g., ["*.safetensors"])
            ignore_patterns: Patterns to exclude

        Returns:
            DownloadResult with download information
        """
        import time

        start_time = time.time()

        try:
            # Get model info first
            model_info = self.client.model_info(model_id, revision=revision)

            # Check if already cached
            cached_path = self._get_cached_path(model_id, revision)
            if cached_path and cached_path.exists():
                logger.info(f"Model {model_id} already cached at {cached_path}")
                return DownloadResult(
                    status=DownloadStatus.CACHED,
                    model_id=model_id,
                    local_path=cached_path,
                    download_time_seconds=0.0,
                )

            # Download model
            logger.info(f"Downloading model {model_id} (revision: {revision})")

            # Determine what to download
            if not allow_patterns:
                # Auto-detect: prefer safetensors over other formats
                allow_patterns = ["*.safetensors", "*.json", "*.txt"]

            downloaded_path = None
            total_bytes = 0

            # Try to download main model file
            for pattern in allow_patterns:
                try:
                    file_path = hf_hub_download(
                        repo_id=model_id,
                        filename=pattern.replace("*", "model"),  # Simplified
                        revision=revision,
                        cache_dir=str(self.cache_dir),
                        token=self.token,
                    )

                    downloaded_path = Path(file_path)
                    if downloaded_path.exists():
                        total_bytes = downloaded_path.stat().st_size
                        break

                except Exception:
                    continue

            if not downloaded_path:
                raise ValueError(f"Failed to download any files for {model_id}")

            download_time = time.time() - start_time

            logger.info(
                f"Downloaded {model_id} to {downloaded_path} "
                f"({total_bytes / 1024 / 1024:.2f} MB in {download_time:.2f}s)"
            )

            return DownloadResult(
                status=DownloadStatus.SUCCESS,
                model_id=model_id,
                local_path=downloaded_path,
                download_time_seconds=download_time,
                downloaded_bytes=total_bytes,
            )

        except HfHubHTTPError as e:
            logger.error(f"HTTP error downloading {model_id}: {e}")
            return DownloadResult(
                status=DownloadStatus.FAILED,
                model_id=model_id,
                error_message=str(e),
            )

        except Exception as e:
            logger.error(f"Failed to download {model_id}: {e}")
            return DownloadResult(
                status=DownloadStatus.FAILED,
                model_id=model_id,
                error_message=str(e),
            )

    def list_cached_models(self) -> list[ModelMetadata]:
        """
        List models cached locally.

        Returns:
            List of cached model metadata
        """
        try:
            cache_info = scan_cache_dir(str(self.cache_dir))

            results = []
            for repo in cache_info.repos:
                try:
                    metadata = ModelMetadata(
                        model_id=repo.repo_id,
                        name=repo.repo_id.split("/")[-1],
                        source=Source.HUGGINGFACE,
                        type=self._infer_model_type(repo.repo_id),
                        base_model=self._infer_base_model(repo.repo_id),
                        size_bytes=repo.size_on_disk,
                        local_path=Path(repo.repo_path),
                    )
                    results.append(metadata)

                except Exception as e:
                    logger.warning(f"Failed to process cached repo {repo.repo_id}: {e}")
                    continue

            logger.info(f"Found {len(results)} cached models")
            return results

        except Exception as e:
            logger.error(f"Failed to list cached models: {e}")
            return []

    def delete_cached_model(self, model_id: str) -> bool:
        """
        Delete model from cache.

        Args:
            model_id: Model ID to delete

        Returns:
            True if deleted successfully
        """
        try:
            # This is a simplified implementation
            # In production, would use huggingface_hub's cache management
            cached_path = self._get_cached_path(model_id)
            if cached_path and cached_path.exists():
                import shutil

                shutil.rmtree(cached_path)
                logger.info(f"Deleted cached model: {model_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to delete cached model {model_id}: {e}")
            return False

    def verify_checksum(self, file_path: Path, expected_hash: str) -> bool:
        """
        Verify file integrity using SHA256.

        Args:
            file_path: Path to file
            expected_hash: Expected SHA256 hash

        Returns:
            True if hash matches
        """
        try:
            sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)

            actual_hash = sha256.hexdigest()
            matches = actual_hash == expected_hash

            if matches:
                logger.info(f"Checksum verified for {file_path}")
            else:
                logger.warning(
                    f"Checksum mismatch for {file_path}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )

            return matches

        except Exception as e:
            logger.error(f"Failed to verify checksum for {file_path}: {e}")
            return False

    def _model_to_metadata(self, model) -> ModelMetadata:
        """Convert HuggingFace ModelInfo to ModelMetadata."""
        return ModelMetadata(
            model_id=model.id,
            name=model.id.split("/")[-1] if "/" in model.id else model.id,
            source=Source.HUGGINGFACE,
            type=self._infer_model_type(model.id),
            base_model=self._infer_base_model(model.id),
            tags=getattr(model, "tags", []),
            description=getattr(model, "description", ""),
            download_count=getattr(model, "downloads", 0),
            remote_url=f"https://huggingface.co/{model.id}",
        )

    def _infer_model_type(self, model_id: str) -> ModelType:
        """Infer model type from ID."""
        model_id_lower = model_id.lower()

        if "lora" in model_id_lower:
            return ModelType.LORA
        elif "vae" in model_id_lower:
            return ModelType.VAE
        elif "controlnet" in model_id_lower:
            return ModelType.CONTROLNET
        elif "embedding" in model_id_lower or "textual" in model_id_lower:
            return ModelType.EMBEDDING
        else:
            return ModelType.BASE_MODEL

    def _infer_base_model(self, model_id: str) -> BaseModel:
        """Infer base model architecture from ID."""
        model_id_lower = model_id.lower()

        if "sdxl" in model_id_lower or "xl" in model_id_lower:
            return BaseModel.SDXL
        elif "sd3" in model_id_lower:
            return BaseModel.SD3
        elif "pony" in model_id_lower:
            return BaseModel.PONY
        elif "flux" in model_id_lower:
            return BaseModel.FLUX
        elif any(v in model_id_lower for v in ["sd-2", "sd2", "v2"]):
            return BaseModel.SD20
        elif any(v in model_id_lower for v in ["sd-1", "sd1", "v1", "1-5", "1.5"]):
            return BaseModel.SD15
        else:
            return BaseModel.UNKNOWN

    def _get_cached_path(
        self, model_id: str, revision: str = "main"
    ) -> Optional[Path]:
        """Get cached path for a model."""
        # Simplified - in production would use huggingface_hub's cache utilities
        cache_path = self.cache_dir / "hub" / f"models--{model_id.replace('/', '--')}"
        if cache_path.exists():
            return cache_path
        return None
