"""CivitAI API integration service."""

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm

from ml_lib.diffusion.intelligent.hub_integration.entities import (
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


class RateLimiter:
    """Simple rate limiter for API requests."""

    def __init__(self, requests_per_second: float = 1.0):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests per second
        """
        self.min_interval = 1.0 / requests_per_second
        self.last_request = 0.0

    def wait_if_needed(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()
        time_since_last = now - self.last_request

        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)

        self.last_request = time.time()


class CivitAIService:
    """Service for interacting with CivitAI API."""

    BASE_URL = "https://civitai.com/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        rate_limit: float = 1.0,
    ):
        """
        Initialize CivitAI service.

        Args:
            api_key: CivitAI API key (optional, for authenticated requests)
            cache_dir: Custom cache directory (default: ~/.cache/civitai)
            rate_limit: Requests per second (default: 1.0)
        """
        self.api_key = api_key
        self.cache_dir = cache_dir or Path.home() / ".cache" / "civitai"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.rate_limiter = RateLimiter(rate_limit)
        self.session = requests.Session()

        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

        logger.info(f"Initialized CivitAI service with cache: {self.cache_dir}")

    def search_models(
        self,
        query: Optional[str] = None,
        type: Optional[ModelType] = None,
        base_model: Optional[BaseModel] = None,
        sort: str = "Highest Rated",
        limit: int = 20,
        filter: Optional[ModelFilter] = None,
    ) -> list[ModelMetadata]:
        """
        Search models on CivitAI.

        Args:
            query: Search query string
            type: Model type filter
            base_model: Base model filter
            sort: Sort option
            limit: Maximum number of results
            filter: Optional additional filters

        Returns:
            List of model metadata
        """
        try:
            params = {"limit": min(limit, 100), "sort": sort}

            if query:
                params["query"] = query

            # Apply type filter
            if type:
                type_map = {
                    ModelType.BASE_MODEL: "Checkpoint",
                    ModelType.LORA: "LORA",
                    ModelType.EMBEDDING: "TextualInversion",
                    ModelType.VAE: "VAE",
                    ModelType.CONTROLNET: "Controlnet",
                }
                if type in type_map:
                    params["types"] = type_map[type]

            # Apply base model filter
            if base_model and base_model != BaseModel.UNKNOWN:
                params["baseModels"] = base_model.value.upper()

            # Add filter params
            if filter:
                params.update(filter.to_civitai_params())

            # Make request
            self.rate_limiter.wait_if_needed()
            response = self.session.get(f"{self.BASE_URL}/models", params=params)
            response.raise_for_status()

            data = response.json()
            items = data.get("items", [])

            results = []
            for item in items:
                try:
                    metadata = self._api_response_to_metadata(item)

                    # Apply additional filters
                    if filter and not filter.matches(metadata):
                        continue

                    results.append(metadata)

                    if len(results) >= limit:
                        break

                except Exception as e:
                    logger.warning(f"Failed to process model {item.get('id')}: {e}")
                    continue

            logger.info(f"Found {len(results)} models on CivitAI")
            return results

        except requests.RequestException as e:
            logger.error(f"Failed to search CivitAI models: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error searching CivitAI: {e}")
            return []

    def get_model_details(self, model_id: int) -> Optional[ModelMetadata]:
        """
        Get detailed information about a model.

        Args:
            model_id: CivitAI model ID

        Returns:
            Model metadata or None if not found
        """
        try:
            self.rate_limiter.wait_if_needed()
            response = self.session.get(f"{self.BASE_URL}/models/{model_id}")
            response.raise_for_status()

            data = response.json()
            metadata = self._api_response_to_metadata(data)

            logger.info(f"Retrieved details for model {model_id}")
            return metadata

        except requests.RequestException as e:
            logger.error(f"Failed to get model {model_id}: {e}")
            return None

    def download_model(
        self,
        model_id: int,
        version_id: Optional[int] = None,
        download_dir: Optional[Path] = None,
    ) -> DownloadResult:
        """
        Download model from CivitAI.

        Args:
            model_id: CivitAI model ID
            version_id: Specific version ID (default: latest)
            download_dir: Download directory (default: cache_dir)

        Returns:
            DownloadResult with download information
        """
        start_time = time.time()

        try:
            # Get model details
            model_data = self.get_model_details(model_id)
            if not model_data:
                return DownloadResult(
                    status=DownloadStatus.FAILED,
                    model_id=str(model_id),
                    error_message="Model not found",
                )

            # Get version info
            self.rate_limiter.wait_if_needed()
            response = self.session.get(f"{self.BASE_URL}/models/{model_id}")
            response.raise_for_status()
            data = response.json()

            versions = data.get("modelVersions", [])
            if not versions:
                return DownloadResult(
                    status=DownloadStatus.FAILED,
                    model_id=str(model_id),
                    error_message="No versions available",
                )

            # Select version
            if version_id:
                version = next((v for v in versions if v["id"] == version_id), None)
            else:
                version = versions[0]  # Latest version

            if not version:
                return DownloadResult(
                    status=DownloadStatus.FAILED,
                    model_id=str(model_id),
                    error_message=f"Version {version_id} not found",
                )

            # Get download URL
            files = version.get("files", [])
            if not files:
                return DownloadResult(
                    status=DownloadStatus.FAILED,
                    model_id=str(model_id),
                    error_message="No files available",
                )

            # Prefer safetensors
            file = next(
                (f for f in files if f.get("name", "").endswith(".safetensors")), files[0]
            )

            download_url = file.get("downloadUrl")
            if not download_url:
                return DownloadResult(
                    status=DownloadStatus.FAILED,
                    model_id=str(model_id),
                    error_message="No download URL available",
                )

            # Setup download
            download_dir = download_dir or self.cache_dir
            download_dir.mkdir(parents=True, exist_ok=True)

            filename = file.get("name", f"model_{model_id}.safetensors")
            file_path = download_dir / filename

            # Check if already exists
            expected_hash = file.get("hashes", {}).get("SHA256", "")
            if file_path.exists() and expected_hash:
                if self.verify_checksum(file_path, expected_hash):
                    logger.info(f"Model already downloaded: {file_path}")
                    return DownloadResult(
                        status=DownloadStatus.CACHED,
                        model_id=str(model_id),
                        local_path=file_path,
                        checksum_verified=True,
                        expected_sha256=expected_hash,
                        actual_sha256=expected_hash,
                    )

            # Download file
            logger.info(f"Downloading {filename} from CivitAI...")
            self.rate_limiter.wait_if_needed()

            response = self.session.get(download_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded_bytes = 0

            with open(file_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=filename,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_bytes += len(chunk)
                            pbar.update(len(chunk))

            download_time = time.time() - start_time

            # Verify checksum
            checksum_ok = False
            actual_hash = ""
            if expected_hash:
                checksum_ok = self.verify_checksum(file_path, expected_hash)
                if checksum_ok:
                    actual_hash = expected_hash
                else:
                    # Calculate actual hash for logging
                    actual_hash = self._calculate_sha256(file_path)

            logger.info(
                f"Downloaded {filename} "
                f"({downloaded_bytes / 1024 / 1024:.2f} MB in {download_time:.2f}s)"
            )

            return DownloadResult(
                status=DownloadStatus.SUCCESS,
                model_id=str(model_id),
                local_path=file_path,
                download_time_seconds=download_time,
                downloaded_bytes=downloaded_bytes,
                checksum_verified=checksum_ok,
                expected_sha256=expected_hash,
                actual_sha256=actual_hash,
            )

        except requests.RequestException as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            return DownloadResult(
                status=DownloadStatus.FAILED,
                model_id=str(model_id),
                error_message=str(e),
            )
        except Exception as e:
            logger.error(f"Unexpected error downloading {model_id}: {e}")
            return DownloadResult(
                status=DownloadStatus.FAILED,
                model_id=str(model_id),
                error_message=str(e),
            )

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
            actual_hash = self._calculate_sha256(file_path)
            matches = actual_hash.lower() == expected_hash.lower()

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

    def _calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _api_response_to_metadata(self, data: dict) -> ModelMetadata:
        """Convert CivitAI API response to ModelMetadata."""
        model_id = str(data.get("id", ""))
        name = data.get("name", "Unknown")

        # Determine model type
        type_str = data.get("type", "Checkpoint")
        type_map = {
            "Checkpoint": ModelType.BASE_MODEL,
            "LORA": ModelType.LORA,
            "TextualInversion": ModelType.EMBEDDING,
            "VAE": ModelType.VAE,
            "Controlnet": ModelType.CONTROLNET,
        }
        model_type = type_map.get(type_str, ModelType.BASE_MODEL)

        # Get latest version info
        versions = data.get("modelVersions", [])
        version_info = versions[0] if versions else {}

        # Infer base model
        base_model_str = version_info.get("baseModel", "").lower()
        base_model = self._parse_base_model(base_model_str)

        # Get tags and trigger words
        tags = data.get("tags", [])
        trigger_words = version_info.get("trainedWords", [])

        # Stats
        stats = data.get("stats", {})
        download_count = stats.get("downloadCount", 0)
        rating = stats.get("rating", 0.0)

        # Files
        files = version_info.get("files", [])
        size_bytes = files[0].get("sizeKB", 0) * 1024 if files else 0

        return ModelMetadata(
            model_id=f"civitai_{model_id}",
            name=name,
            source=Source.CIVITAI,
            type=model_type,
            base_model=base_model,
            version=str(version_info.get("id", "")),
            format=ModelFormat.SAFETENSORS,  # Most CivitAI models use safetensors
            size_bytes=size_bytes,
            trigger_words=trigger_words,
            tags=tags,
            description=data.get("description", ""),
            download_count=download_count,
            rating=rating,
            remote_url=f"https://civitai.com/models/{model_id}",
        )

    def _parse_base_model(self, base_model_str: str) -> BaseModel:
        """Parse base model from string."""
        base_model_str = base_model_str.lower()

        if "xl" in base_model_str or "sdxl" in base_model_str:
            return BaseModel.SDXL
        elif "pony" in base_model_str:
            return BaseModel.PONY
        elif "sd 3" in base_model_str or "sd3" in base_model_str:
            return BaseModel.SD3
        elif "sd 2" in base_model_str or "sd2" in base_model_str:
            return BaseModel.SD20
        elif "sd 1" in base_model_str or "sd1" in base_model_str or "1.5" in base_model_str:
            return BaseModel.SD15
        else:
            return BaseModel.UNKNOWN
