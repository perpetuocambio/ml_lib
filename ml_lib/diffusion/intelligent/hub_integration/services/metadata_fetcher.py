"""
Secure and Anonymous Metadata Fetcher.

Downloads model metadata from CivitAI and HuggingFace securely:
- No API keys stored
- Anonymous requests
- Privacy-focused
- Rate limiting built-in
- Local caching

Provides our own format, independent of ComfyUI custom_nodes.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """
    Our own metadata format.

    Independent from ComfyUI JSON format.
    """

    # Identity
    model_id: str  # Unique ID (hash or CivitAI ID)
    name: str
    version: str = "main"
    source: str = "local"  # "local", "civitai", "huggingface"

    # File info
    file_path: Optional[Path] = None
    file_size_bytes: int = 0
    file_hash_sha256: str = ""

    # Model classification
    base_architecture: str = "unknown"  # "SDXL", "SD15", "Flux", etc.
    model_type: str = "unknown"  # "checkpoint", "lora", "vae", etc.

    # Quality indicators (for selection)
    download_count: int = 0
    rating: float = 0.0
    favorites: int = 0

    # Usage hints
    trigger_words: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    description: str = ""

    # Optimal parameters (from example images)
    optimal_steps: Optional[int] = None
    optimal_cfg: Optional[float] = None
    optimal_sampler: Optional[str] = None
    optimal_scheduler: Optional[str] = None
    optimal_clip_skip: Optional[int] = None
    optimal_lora_weight: float = 1.0

    # Privacy-safe metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Internal
    _last_fetched: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "file_path": str(self.file_path) if self.file_path else None,
            "file_size_bytes": self.file_size_bytes,
            "file_hash_sha256": self.file_hash_sha256,
            "base_architecture": self.base_architecture,
            "model_type": self.model_type,
            "download_count": self.download_count,
            "rating": self.rating,
            "favorites": self.favorites,
            "trigger_words": self.trigger_words,
            "tags": self.tags,
            "description": self.description,
            "optimal_steps": self.optimal_steps,
            "optimal_cfg": self.optimal_cfg,
            "optimal_sampler": self.optimal_sampler,
            "optimal_scheduler": self.optimal_scheduler,
            "optimal_clip_skip": self.optimal_clip_skip,
            "optimal_lora_weight": self.optimal_lora_weight,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "_last_fetched": self._last_fetched.isoformat() if self._last_fetched else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelMetadata":
        """Load from dict."""
        # Parse dates
        if data.get("created_at"):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if data.get("_last_fetched"):
            data["_last_fetched"] = datetime.fromisoformat(data["_last_fetched"])

        # Parse path
        if data.get("file_path"):
            data["file_path"] = Path(data["file_path"])

        return cls(**data)


@dataclass
class FetcherConfig:
    """Configuration for metadata fetcher."""

    # Cache settings
    cache_dir: Path = Path.home() / ".ml_lib" / "metadata_cache"
    cache_ttl_hours: int = 24  # Refresh after 24h

    # Rate limiting (be respectful to APIs)
    min_request_interval_seconds: float = 1.0  # Wait 1s between requests
    max_retries: int = 3
    request_timeout_seconds: int = 30

    # Privacy
    use_anonymous_requests: bool = True  # No API keys, no user tracking
    user_agent: str = "ml_lib/1.0"  # Generic user agent

    # CivitAI API
    civitai_api_base: str = "https://civitai.com/api/v1"

    def __post_init__(self):
        """Create cache directory."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class MetadataFetcher:
    """
    Secure and anonymous metadata fetcher.

    Features:
    - No API keys required
    - Anonymous requests (privacy-first)
    - Local caching
    - Rate limiting (respectful)
    - Our own format (independent)

    Example:
        >>> fetcher = MetadataFetcher()
        >>>
        >>> # From file hash
        >>> metadata = fetcher.fetch_by_hash("af7ed3e1fc3794bb...")
        >>>
        >>> # From file path
        >>> metadata = fetcher.fetch_for_file("/models/lora.safetensors")
        >>>
        >>> # Bulk update
        >>> fetcher.update_directory("/models/loras")
    """

    def __init__(self, config: Optional[FetcherConfig] = None):
        """
        Initialize fetcher.

        Args:
            config: Fetcher configuration (None = defaults)
        """
        self.config = config or FetcherConfig()
        self._last_request_time: float = 0.0

        logger.info(f"MetadataFetcher initialized (cache: {self.config.cache_dir})")

    def fetch_for_file(
        self, file_path: Path | str, force_refresh: bool = False
    ) -> Optional[ModelMetadata]:
        """
        Fetch metadata for a model file.

        Args:
            file_path: Path to model file
            force_refresh: Ignore cache and fetch fresh

        Returns:
            Model metadata or None if not found
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None

        # Calculate file hash (for lookup)
        file_hash = self._calculate_file_hash(file_path)

        # Check cache first
        if not force_refresh:
            cached = self._load_from_cache(file_hash)
            if cached and self._is_cache_valid(cached):
                logger.debug(f"Using cached metadata for {file_path.name}")
                cached.file_path = file_path  # Update path
                return cached

        # Fetch from CivitAI by hash
        logger.info(f"Fetching metadata for {file_path.name}...")
        metadata = self._fetch_from_civitai_by_hash(file_hash)

        if metadata:
            metadata.file_path = file_path
            metadata.file_size_bytes = file_path.stat().st_size
            metadata.file_hash_sha256 = file_hash
            metadata._last_fetched = datetime.now()

            # Save to cache
            self._save_to_cache(file_hash, metadata)

            return metadata

        # Fallback: create basic metadata from file
        logger.debug(f"No online metadata found for {file_path.name}, using file info")
        return self._create_basic_metadata(file_path, file_hash)

    def fetch_by_hash(
        self, sha256_hash: str, force_refresh: bool = False
    ) -> Optional[ModelMetadata]:
        """
        Fetch metadata by file hash.

        Args:
            sha256_hash: SHA256 hash of model file
            force_refresh: Ignore cache

        Returns:
            Model metadata or None
        """
        # Check cache
        if not force_refresh:
            cached = self._load_from_cache(sha256_hash)
            if cached and self._is_cache_valid(cached):
                return cached

        # Fetch from CivitAI
        metadata = self._fetch_from_civitai_by_hash(sha256_hash)

        if metadata:
            metadata._last_fetched = datetime.now()
            self._save_to_cache(sha256_hash, metadata)

        return metadata

    def update_directory(
        self,
        directory: Path | str,
        extensions: Optional[list[str]] = None,
        force_refresh: bool = False,
    ) -> dict[Path, Optional[ModelMetadata]]:
        """
        Update metadata for all models in directory.

        Args:
            directory: Directory to scan
            extensions: File extensions to include (None = all common)
            force_refresh: Force refresh cached metadata

        Returns:
            Dict mapping file paths to metadata
        """
        directory = Path(directory)

        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return {}

        if extensions is None:
            extensions = ["safetensors", "pt", "ckpt", "pth"]

        # Find all model files
        model_files = []
        for ext in extensions:
            model_files.extend(directory.rglob(f"*.{ext}"))

        logger.info(f"Updating metadata for {len(model_files)} models in {directory}...")

        results = {}
        for file_path in model_files:
            metadata = self.fetch_for_file(file_path, force_refresh=force_refresh)
            results[file_path] = metadata

            # Show progress
            if len(results) % 10 == 0:
                logger.info(f"Progress: {len(results)}/{len(model_files)}")

        successful = sum(1 for m in results.values() if m is not None)
        logger.info(f"Updated {successful}/{len(model_files)} models")

        return results

    def _calculate_file_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Calculate SHA256 hash of file."""
        hasher = hashlib.sha256()

        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)

        return hasher.hexdigest().upper()

    def _fetch_from_civitai_by_hash(self, sha256_hash: str) -> Optional[ModelMetadata]:
        """
        Fetch metadata from CivitAI API using file hash.

        CivitAI provides a public API to lookup models by hash.
        No API key needed for basic queries.
        """
        try:
            import requests

            # Rate limiting
            self._wait_for_rate_limit()

            # CivitAI hash lookup endpoint
            url = f"{self.config.civitai_api_base}/model-versions/by-hash/{sha256_hash}"

            headers = {"User-Agent": self.config.user_agent}

            # Anonymous request (no API key)
            response = requests.get(
                url, headers=headers, timeout=self.config.request_timeout_seconds
            )

            self._last_request_time = time.time()

            if response.status_code == 404:
                logger.debug(f"Model not found on CivitAI: {sha256_hash[:8]}...")
                return None

            if response.status_code != 200:
                logger.warning(f"CivitAI API error: {response.status_code}")
                return None

            data = response.json()

            # Parse CivitAI response to our format
            return self._parse_civitai_response(data)

        except Exception as e:
            logger.warning(f"Failed to fetch from CivitAI: {e}")
            return None

    def _parse_civitai_response(self, data: dict) -> ModelMetadata:
        """Parse CivitAI API response to our format."""
        # Extract model info
        model_info = data.get("model", {})

        model_id = str(data.get("id", ""))
        name = model_info.get("name", "Unknown")
        version = data.get("name", "main")
        base_model = data.get("baseModel", "unknown")
        model_type = model_info.get("type", "unknown")

        # Stats
        stats = data.get("stats", {})
        download_count = stats.get("downloadCount", 0)
        rating = stats.get("rating", 0.0)
        favorites = stats.get("favoriteCount", 0)

        # Trigger words
        trigger_words = data.get("trainedWords", [])

        # Tags (from model)
        tags = [tag.get("name", "") for tag in model_info.get("tags", [])]

        # Description
        description = data.get("description", "")

        # Extract optimal parameters from first image
        images = data.get("images", [])
        optimal_steps = None
        optimal_cfg = None
        optimal_sampler = None
        optimal_scheduler = None
        optimal_clip_skip = None
        optimal_lora_weight = 1.0

        if images:
            meta = images[0].get("meta", {})
            optimal_steps = meta.get("steps")
            optimal_cfg = meta.get("cfgScale")
            optimal_sampler = meta.get("sampler")
            optimal_scheduler = meta.get("Scheduler")
            optimal_clip_skip = meta.get("clipSkip")

            # LoRA weight
            lora_weights = meta.get("Lora weights", {})
            if lora_weights:
                optimal_lora_weight = float(list(lora_weights.values())[0])

        # Timestamps
        created_at = None
        updated_at = None
        if data.get("createdAt"):
            created_at = datetime.fromisoformat(
                data["createdAt"].replace("Z", "+00:00")
            )
        if data.get("updatedAt"):
            updated_at = datetime.fromisoformat(
                data["updatedAt"].replace("Z", "+00:00")
            )

        return ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            source="civitai",
            base_architecture=base_model,
            model_type=model_type.lower(),
            download_count=download_count,
            rating=rating,
            favorites=favorites,
            trigger_words=trigger_words,
            tags=tags,
            description=description[:500],  # Limit description length
            optimal_steps=optimal_steps,
            optimal_cfg=optimal_cfg,
            optimal_sampler=optimal_sampler,
            optimal_scheduler=optimal_scheduler,
            optimal_clip_skip=optimal_clip_skip,
            optimal_lora_weight=optimal_lora_weight,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _create_basic_metadata(self, file_path: Path, file_hash: str) -> ModelMetadata:
        """Create basic metadata from file when no online data available."""
        return ModelMetadata(
            model_id=file_hash[:16],
            name=file_path.stem,
            source="local",
            file_path=file_path,
            file_size_bytes=file_path.stat().st_size,
            file_hash_sha256=file_hash,
            base_architecture=self._guess_architecture_from_filename(file_path.name),
            model_type=self._guess_type_from_path(file_path),
            _last_fetched=datetime.now(),
        )

    def _guess_architecture_from_filename(self, filename: str) -> str:
        """Guess architecture from filename."""
        filename_lower = filename.lower()

        if "sdxl" in filename_lower or "xl" in filename_lower:
            return "SDXL"
        elif "flux" in filename_lower:
            return "Flux"
        elif "sd3" in filename_lower:
            return "SD3"
        elif "sd15" in filename_lower or "sd_v15" in filename_lower:
            return "SD15"
        elif "pony" in filename_lower:
            return "Pony"
        else:
            return "unknown"

    def _guess_type_from_path(self, file_path: Path) -> str:
        """Guess model type from directory structure."""
        path_str = str(file_path).lower()

        if "/lora" in path_str:
            return "lora"
        elif "/checkpoint" in path_str:
            return "checkpoint"
        elif "/vae" in path_str:
            return "vae"
        elif "/controlnet" in path_str:
            return "controlnet"
        elif "/embedding" in path_str:
            return "embedding"
        else:
            return "unknown"

    def _wait_for_rate_limit(self):
        """Wait to respect rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.min_request_interval_seconds:
            sleep_time = self.config.min_request_interval_seconds - elapsed
            time.sleep(sleep_time)

    def _get_cache_path(self, file_hash: str) -> Path:
        """Get cache file path for hash."""
        return self.config.cache_dir / f"{file_hash[:8]}.json"

    def _save_to_cache(self, file_hash: str, metadata: ModelMetadata):
        """Save metadata to cache."""
        try:
            cache_path = self._get_cache_path(file_hash)
            with open(cache_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_from_cache(self, file_hash: str) -> Optional[ModelMetadata]:
        """Load metadata from cache."""
        try:
            cache_path = self._get_cache_path(file_hash)
            if not cache_path.exists():
                return None

            with open(cache_path) as f:
                data = json.load(f)

            return ModelMetadata.from_dict(data)
        except Exception as e:
            logger.debug(f"Failed to load cache: {e}")
            return None

    def _is_cache_valid(self, metadata: ModelMetadata) -> bool:
        """Check if cached metadata is still valid."""
        if metadata._last_fetched is None:
            return False

        age = datetime.now() - metadata._last_fetched
        max_age = timedelta(hours=self.config.cache_ttl_hours)

        return age < max_age
