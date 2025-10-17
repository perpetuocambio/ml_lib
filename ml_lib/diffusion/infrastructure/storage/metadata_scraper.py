"""
Intelligent metadata scraper for updating model information from APIs.

Fetches missing metadata from CivitAI/HuggingFace and respects user overrides.
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

from ml_lib.diffusion.model_enums import ModelType, BaseModel, Source
from ml_lib.diffusion.model_metadata import ModelMetadata
from ml_lib.diffusion.infrastructure.storage.metadata_db import MetadataDatabase

logger = logging.getLogger(__name__)


class MetadataScraper:
    """
    Intelligent metadata scraper that:
    - Only fetches models that don't exist in database
    - Updates stale metadata (>30 days old)
    - Respects user overrides (never overwrites user customizations)
    - Batches API requests efficiently
    """

    def __init__(
        self,
        db: MetadataDatabase,
        refresh_days: int = 30,
        batch_size: int = 50,
    ):
        """
        Initialize scraper.

        Args:
            db: MetadataDatabase instance
            refresh_days: Refresh metadata older than N days
            batch_size: Number of models to process per batch
        """
        self.db = db
        self.refresh_threshold = timedelta(days=refresh_days)
        self.batch_size = batch_size

    def sync_from_civitai(
        self,
        model_ids: Optional[list[str]] = None,
        force_refresh: bool = False,
    ) -> dict[str, int]:
        """
        Sync metadata from CivitAI.

        Args:
            model_ids: Specific model IDs to sync (None = all)
            force_refresh: Force refresh even if not stale

        Returns:
            Dict with counts: {"added": N, "updated": N, "skipped": N, "failed": N}
        """
        results = {"added": 0, "updated": 0, "skipped": 0, "failed": 0}

        if model_ids:
            # Specific models requested
            models_to_fetch = model_ids
        else:
            # Find models that need updating
            models_to_fetch = self._find_stale_models(Source.CIVITAI, force_refresh)

        logger.info(f"Syncing {len(models_to_fetch)} models from CivitAI")

        # Process in batches
        for i in range(0, len(models_to_fetch), self.batch_size):
            batch = models_to_fetch[i:i + self.batch_size]
            batch_results = self._process_civitai_batch(batch, force_refresh)

            # Aggregate results
            for key in results:
                results[key] += batch_results.get(key, 0)

            logger.info(
                f"Batch {i//self.batch_size + 1}: "
                f"+{batch_results['added']} ~{batch_results['updated']} "
                f"-{batch_results['skipped']} !{batch_results['failed']}"
            )

        return results

    def sync_from_huggingface(
        self,
        model_ids: Optional[list[str]] = None,
        force_refresh: bool = False,
    ) -> dict[str, int]:
        """
        Sync metadata from HuggingFace.

        Args:
            model_ids: Specific model IDs to sync (None = all)
            force_refresh: Force refresh even if not stale

        Returns:
            Dict with counts: {"added": N, "updated": N, "skipped": N, "failed": N}
        """
        results = {"added": 0, "updated": 0, "skipped": 0, "failed": 0}

        if model_ids:
            models_to_fetch = model_ids
        else:
            models_to_fetch = self._find_stale_models(Source.HUGGINGFACE, force_refresh)

        logger.info(f"Syncing {len(models_to_fetch)} models from HuggingFace")

        # Process in batches
        for i in range(0, len(models_to_fetch), self.batch_size):
            batch = models_to_fetch[i:i + self.batch_size]
            batch_results = self._process_huggingface_batch(batch, force_refresh)

            for key in results:
                results[key] += batch_results.get(key, 0)

            logger.info(
                f"Batch {i//self.batch_size + 1}: "
                f"+{batch_results['added']} ~{batch_results['updated']} "
                f"-{batch_results['skipped']} !{batch_results['failed']}"
            )

        return results

    def sync_local_models(self, models_dir: Path, model_type: ModelType) -> dict[str, int]:
        """
        Sync local models (files without metadata).

        Scans directory and adds entries for models not in database.

        Args:
            models_dir: Directory containing models
            model_type: Type of models in directory

        Returns:
            Dict with counts
        """
        results = {"added": 0, "skipped": 0}

        if not models_dir.exists():
            logger.warning(f"Directory not found: {models_dir}")
            return results

        # Find model files
        extensions = [".safetensors", ".ckpt", ".pt", ".pth"]
        model_files = []

        for ext in extensions:
            model_files.extend(models_dir.rglob(f"*{ext}"))

        logger.info(f"Found {len(model_files)} model files in {models_dir}")

        for model_file in model_files:
            try:
                # Check if already in database (by local_path)
                existing = self._get_by_local_path(model_file)

                if existing:
                    results["skipped"] += 1
                    continue

                # Create minimal metadata
                metadata = self._create_local_metadata(model_file, model_type)

                if self.db.insert_model(metadata):
                    results["added"] += 1
                    logger.debug(f"Added local model: {model_file.name}")
                else:
                    logger.warning(f"Failed to add: {model_file.name}")

            except Exception as e:
                logger.error(f"Error processing {model_file}: {e}")

        logger.info(f"Local sync complete: {results['added']} added, {results['skipped']} skipped")
        return results

    def _find_stale_models(self, source: Source, force_refresh: bool) -> list[str]:
        """
        Find models that need metadata refresh.

        Args:
            source: Source to check (CIVITAI or HUGGINGFACE)
            force_refresh: Include all models regardless of age

        Returns:
            List of model IDs to fetch
        """
        # TODO: Implement query for stale models
        # For now, return empty list
        return []

    def _process_civitai_batch(
        self,
        model_ids: list[str],
        force_refresh: bool,
    ) -> dict[str, int]:
        """
        Process a batch of CivitAI models.

        Args:
            model_ids: Model IDs to fetch
            force_refresh: Force update existing

        Returns:
            Batch results
        """
        results = {"added": 0, "updated": 0, "skipped": 0, "failed": 0}

        # TODO: Implement CivitAI API fetching
        # Import and use ml_lib.diffusion.services.civitai_service
        # Check for user overrides before updating

        return results

    def _process_huggingface_batch(
        self,
        model_ids: list[str],
        force_refresh: bool,
    ) -> dict[str, int]:
        """
        Process a batch of HuggingFace models.

        Args:
            model_ids: Model IDs to fetch
            force_refresh: Force update existing

        Returns:
            Batch results
        """
        results = {"added": 0, "updated": 0, "skipped": 0, "failed": 0}

        # TODO: Implement HuggingFace API fetching
        # Import and use ml_lib.diffusion.services.huggingface_service
        # Check for user overrides before updating

        return results

    def _get_by_local_path(self, local_path: Path):
        """Get model by local path."""
        # Query database for model with matching local_path
        try:
            rows = self.db.execute(
                "SELECT model_id FROM models WHERE local_path = ?",
                (str(local_path),)
            )
            if rows:
                return self.db.get_model(rows[0]["model_id"])
            return None
        except Exception as e:
            logger.error(f"Failed to query by path: {e}")
            return None

    def _create_local_metadata(
        self,
        model_file: Path,
        model_type: ModelType,
    ) -> ModelMetadata:
        """
        Create minimal metadata for local model file.

        Args:
            model_file: Path to model file
            model_type: Type of model

        Returns:
            ModelMetadata with basic info
        """
        # Generate model ID from file name
        model_id = f"local_{model_file.stem}"

        # Detect base model from filename
        base_model = self._detect_base_model(model_file.name)

        # Get file size
        size_bytes = model_file.stat().st_size if model_file.exists() else 0

        # Determine format
        format_map = {
            ".safetensors": "safetensors",
            ".ckpt": "ckpt",
            ".pt": "pickle",
            ".pth": "pickle",
        }
        from ml_lib.diffusion.model_enums import ModelFormat
        model_format = ModelFormat(format_map.get(model_file.suffix.lower(), "safetensors"))

        return ModelMetadata(
            model_id=model_id,
            name=model_file.stem,
            source=Source.LOCAL,
            type=model_type,
            base_model=base_model,
            version="local",
            format=model_format,
            size_bytes=size_bytes,
            sha256="",
            trigger_words=[],
            tags=[],
            description="",
            download_count=0,
            rating=0.0,
            recommended_weight=None,
            local_path=model_file,
            remote_url="",
        )

    def _detect_base_model(self, filename: str) -> BaseModel:
        """
        Detect base model from filename.

        Args:
            filename: Model filename

        Returns:
            BaseModel enum
        """
        filename_lower = filename.lower()

        if "pony" in filename_lower:
            return BaseModel.PONY
        elif "sdxl" in filename_lower or "xl" in filename_lower:
            return BaseModel.SDXL
        elif "sd3" in filename_lower:
            return BaseModel.SD3
        elif "flux" in filename_lower:
            return BaseModel.FLUX
        elif "sd15" in filename_lower or "sd_1.5" in filename_lower:
            return BaseModel.SD15
        elif "sd21" in filename_lower or "sd_2.1" in filename_lower:
            return BaseModel.SD21
        elif "sd20" in filename_lower or "sd_2.0" in filename_lower:
            return BaseModel.SD20
        else:
            return BaseModel.UNKNOWN
