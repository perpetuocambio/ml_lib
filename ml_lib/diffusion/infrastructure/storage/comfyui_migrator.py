"""
Migrate model metadata from ComfyUI metadata.json files to SQLite database.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from ml_lib.diffusion.model_enums import (
    ModelType,
    BaseModel,
    Source,
    ModelFormat,
)
from ml_lib.diffusion.storage.metadata_db import MetadataDatabase

logger = logging.getLogger(__name__)


class ComfyUIMetadataMigrator:
    """
    Migrate metadata from ComfyUI's metadata.json format to SQLite.

    Supports scanning ComfyUI model directories and importing all metadata.
    """

    def __init__(self, db: MetadataDatabase):
        """
        Initialize migrator.

        Args:
            db: Target MetadataDatabase instance
        """
        self.db = db

    def migrate_directory(
        self,
        models_dir: Path,
        model_type: ModelType,
        recursive: bool = True,
    ) -> tuple[int, int]:
        """
        Migrate all metadata.json files in a directory.

        Args:
            models_dir: Directory containing models
            model_type: Type of models in this directory
            recursive: Search recursively

        Returns:
            Tuple of (successful, failed) counts
        """
        if not models_dir.exists():
            logger.warning(f"Directory not found: {models_dir}")
            return (0, 0)

        logger.info(f"Migrating {model_type.value} models from {models_dir}")

        pattern = "**/*.metadata.json" if recursive else "*.metadata.json"
        metadata_files = list(models_dir.glob(pattern))

        successful = 0
        failed = 0

        for metadata_file in metadata_files:
            try:
                # Find corresponding model file
                model_file = self._find_model_file(metadata_file)

                if not model_file:
                    logger.warning(f"No model file found for {metadata_file}")
                    failed += 1
                    continue

                # Parse and insert
                metadata = self._parse_comfyui_metadata(
                    metadata_file,
                    model_file,
                    model_type
                )

                if metadata and self.db.insert_model(metadata):
                    successful += 1
                    logger.debug(f"Migrated: {metadata.name}")
                else:
                    failed += 1

            except Exception as e:
                logger.error(f"Failed to migrate {metadata_file}: {e}")
                failed += 1

        logger.info(f"Migration complete: {successful} successful, {failed} failed")
        return (successful, failed)

    def _find_model_file(self, metadata_file: Path) -> Optional[Path]:
        """
        Find the model file corresponding to a metadata.json file.

        Args:
            metadata_file: Path to .metadata.json file

        Returns:
            Path to model file or None
        """
        # Remove .metadata.json extension
        base_name = metadata_file.name.replace(".metadata.json", "")

        # Common model extensions
        extensions = [".safetensors", ".ckpt", ".pt", ".pth", ".bin"]

        for ext in extensions:
            model_file = metadata_file.parent / f"{base_name}{ext}"
            if model_file.exists():
                return model_file

        return None

    def _parse_comfyui_metadata(
        self,
        metadata_file: Path,
        model_file: Path,
        model_type: ModelType,
    ):
        """
        Parse ComfyUI metadata.json format.

        Args:
            metadata_file: Path to metadata.json
            model_file: Path to model file
            model_type: Model type

        Returns:
            ModelMetadata or None if parsing fails
        """
        # Import here to avoid circular import
        from ml_lib.diffusion.model_metadata import ModelMetadata

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # ComfyUI metadata structure varies, handle different formats
            # Common structure: {"modelId": "xxx", "name": "xxx", ...}

            # Extract base info
            model_id = str(data.get("modelId") or data.get("id") or model_file.stem)
            name = data.get("name") or model_file.stem

            # Detect base model
            base_model_str = data.get("baseModel", "").lower()
            base_model = self._parse_base_model(base_model_str, name)

            # Get tags (may be in different fields)
            tags = []
            if "tags" in data and isinstance(data["tags"], list):
                tags = data["tags"]
            elif "trainedWords" in data:
                # Sometimes tags are in trainedWords
                trained = data["trainedWords"]
                if isinstance(trained, list):
                    tags = trained

            # Get trigger words
            trigger_words = []
            if "triggerWords" in data and isinstance(data["triggerWords"], list):
                trigger_words = data["triggerWords"]
            elif "trainedWords" in data and isinstance(data["trainedWords"], list):
                trigger_words = data["trainedWords"]

            # Get stats
            download_count = data.get("stats", {}).get("downloadCount", 0)
            rating = data.get("stats", {}).get("rating", 0.0) or 0.0

            # For LoRAs, get recommended weight
            recommended_weight = None
            if model_type == ModelType.LORA:
                recommended_weight = data.get("recommendedWeight") or 1.0

            # Get description
            description = data.get("description", "")

            # Get file info
            size_bytes = model_file.stat().st_size if model_file.exists() else 0
            sha256 = data.get("files", [{}])[0].get("hashes", {}).get("SHA256", "")

            # Determine format
            format_str = model_file.suffix.lower()
            format_map = {
                ".safetensors": ModelFormat.SAFETENSORS,
                ".ckpt": ModelFormat.CKPT,
                ".pt": ModelFormat.PICKLE,
                ".pth": ModelFormat.PICKLE,
            }
            model_format = format_map.get(format_str, ModelFormat.SAFETENSORS)

            return ModelMetadata(
                model_id=model_id,
                name=name,
                source=Source.CIVITAI,  # Most ComfyUI metadata is from CivitAI
                type=model_type,
                base_model=base_model,
                version="main",
                format=model_format,
                size_bytes=size_bytes,
                sha256=sha256,
                trigger_words=trigger_words,
                tags=tags,
                description=description,
                download_count=download_count,
                rating=rating,
                recommended_weight=recommended_weight,
                local_path=model_file,
                remote_url="",
            )

        except Exception as e:
            logger.error(f"Failed to parse {metadata_file}: {e}")
            return None

    def _parse_base_model(self, base_model_str: str, name: str = "") -> BaseModel:
        """
        Parse base model from string or name.

        Args:
            base_model_str: Base model string from metadata
            name: Model name (fallback detection)

        Returns:
            BaseModel enum
        """
        combined = f"{base_model_str} {name}".lower()

        if "pony" in combined:
            return BaseModel.PONY
        elif "sdxl" in combined or "xl" in combined:
            return BaseModel.SDXL
        elif "sd3" in combined:
            return BaseModel.SD3
        elif "flux" in combined:
            return BaseModel.FLUX
        elif "sd 1.5" in combined or "sd15" in combined or "sd 2" in combined:
            # SD 1.5 or 2.x
            if "2.1" in combined:
                return BaseModel.SD21
            elif "2.0" in combined:
                return BaseModel.SD20
            else:
                return BaseModel.SD15
        else:
            return BaseModel.UNKNOWN

    def migrate_comfyui_installation(
        self,
        comfyui_root: Path,
    ) -> dict[str, tuple[int, int]]:
        """
        Migrate entire ComfyUI installation.

        Args:
            comfyui_root: Root directory of ComfyUI

        Returns:
            Dict mapping model type to (successful, failed) counts
        """
        results = {}

        # Standard ComfyUI directory structure
        migrations = [
            (comfyui_root / "models" / "checkpoints", ModelType.BASE_MODEL),
            (comfyui_root / "models" / "loras", ModelType.LORA),
            (comfyui_root / "models" / "vae", ModelType.VAE),
            (comfyui_root / "models" / "controlnet", ModelType.CONTROLNET),
            (comfyui_root / "models" / "embeddings", ModelType.EMBEDDING),
            (comfyui_root / "models" / "ipadapter", ModelType.IPADAPTER),
        ]

        for models_dir, model_type in migrations:
            if models_dir.exists():
                results[model_type.value] = self.migrate_directory(models_dir, model_type)

        return results
