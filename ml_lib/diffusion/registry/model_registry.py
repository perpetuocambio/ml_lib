"""Unified model registry with SQLite persistence."""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from ml_lib.diffusion.models import (
    ModelMetadata,
    ModelFilter,
    Source,
    ModelType,
    BaseModel,
)
from ml_lib.diffusion.comfyui.huggingface_service import (
    HuggingFaceHubService,
)
from ml_lib.diffusion.comfyui.civitai_service import (
    CivitAIService,
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Unified registry for models from all sources."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        hf_service: Optional[HuggingFaceHubService] = None,
        civitai_service: Optional[CivitAIService] = None,
    ):
        """
        Initialize model registry.

        Args:
            db_path: Path to SQLite database (default: ~/.ml_lib/models.db)
            hf_service: HuggingFace service instance
            civitai_service: CivitAI service instance
        """
        self.db_path = db_path or Path.home() / ".ml_lib" / "models.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.hf_service = hf_service or HuggingFaceHubService()
        self.civitai_service = civitai_service or CivitAIService()

        # Initialize database
        self._init_db()

        logger.info(f"Initialized ModelRegistry with database: {self.db_path}")

    def _init_db(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    type TEXT NOT NULL,
                    base_model TEXT NOT NULL,
                    version TEXT,
                    format TEXT NOT NULL,
                    size_bytes INTEGER,
                    sha256 TEXT,
                    trigger_words TEXT,
                    tags TEXT,
                    description TEXT,
                    download_count INTEGER,
                    rating REAL,
                    recommended_weight REAL,
                    local_path TEXT,
                    remote_url TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON models(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON models(type)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_base_model ON models(base_model)"
            )

            conn.commit()

        logger.info("Database schema initialized")

    def register_model(self, metadata: ModelMetadata) -> None:
        """
        Register a model in the registry.

        Args:
            metadata: Model metadata to register
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            # Serialize lists to JSON
            trigger_words_json = json.dumps(metadata.trigger_words)
            tags_json = json.dumps(metadata.tags)

            conn.execute(
                """
                INSERT OR REPLACE INTO models (
                    model_id, name, source, type, base_model, version, format,
                    size_bytes, sha256, trigger_words, tags, description,
                    download_count, rating, recommended_weight, local_path,
                    remote_url, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.model_id,
                    metadata.name,
                    metadata.source.value,
                    metadata.type.value,
                    metadata.base_model.value,
                    metadata.version,
                    metadata.format.value,
                    metadata.size_bytes,
                    metadata.sha256,
                    trigger_words_json,
                    tags_json,
                    metadata.description,
                    metadata.download_count,
                    metadata.rating,
                    metadata.recommended_weight,
                    str(metadata.local_path) if metadata.local_path else None,
                    metadata.remote_url,
                    metadata.created_at.isoformat(),
                    metadata.updated_at.isoformat(),
                ),
            )
            conn.commit()

        logger.info(f"Registered model: {metadata.model_id}")

    def register_from_huggingface(
        self, model_id: str, download: bool = True
    ) -> ModelMetadata:
        """
        Register model from HuggingFace.

        Args:
            model_id: HuggingFace model ID
            download: Whether to download the model

        Returns:
            Registered model metadata
        """
        # Search for the model
        results = self.hf_service.search_models(model_id, limit=1)
        if not results:
            raise ValueError(f"Model not found on HuggingFace: {model_id}")

        metadata = results[0]

        # Download if requested
        if download:
            result = self.hf_service.download_model(model_id)
            if result.success:
                metadata.local_path = result.local_path
                metadata.sha256 = result.actual_sha256

        # Register
        self.register_model(metadata)

        logger.info(f"Registered HuggingFace model: {model_id}")
        return metadata

    def register_from_civitai(
        self, model_id: int, download: bool = True
    ) -> ModelMetadata:
        """
        Register model from CivitAI.

        Args:
            model_id: CivitAI model ID
            download: Whether to download the model

        Returns:
            Registered model metadata
        """
        # Get model details
        metadata = self.civitai_service.get_model_details(model_id)
        if not metadata:
            raise ValueError(f"Model not found on CivitAI: {model_id}")

        # Download if requested
        if download:
            result = self.civitai_service.download_model(model_id)
            if result.success:
                metadata.local_path = result.local_path
                metadata.sha256 = result.actual_sha256

        # Register
        self.register_model(metadata)

        logger.info(f"Registered CivitAI model: {model_id}")
        return metadata

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Get model metadata by ID.

        Args:
            model_id: Model ID

        Returns:
            Model metadata or None if not found
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM models WHERE model_id = ?", (model_id,)
            )
            row = cursor.fetchone()

            if row:
                return self._row_to_metadata(row)
            return None

    def search(
        self,
        query: Optional[str] = None,
        sources: Optional[list[Source]] = None,
        model_type: Optional[ModelType] = None,
        base_model: Optional[BaseModel] = None,
        limit: int = 20,
    ) -> list[ModelMetadata]:
        """
        Search models in registry.

        Args:
            query: Text search query
            sources: Filter by sources
            model_type: Filter by model type
            base_model: Filter by base model
            limit: Maximum results

        Returns:
            List of matching models
        """
        conditions = []
        params = []

        if query:
            conditions.append("(name LIKE ? OR description LIKE ? OR tags LIKE ?)")
            search_term = f"%{query}%"
            params.extend([search_term, search_term, search_term])

        if sources:
            placeholders = ",".join("?" * len(sources))
            conditions.append(f"source IN ({placeholders})")
            params.extend([s.value for s in sources])

        if model_type:
            conditions.append("type = ?")
            params.append(model_type.value)

        if base_model:
            conditions.append("base_model = ?")
            params.append(base_model.value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM models WHERE {where_clause} LIMIT ?"
        params.append(limit)

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

            return [self._row_to_metadata(row) for row in rows]

    def list_models(
        self,
        source: Optional[Source] = None,
        model_type: Optional[ModelType] = None,
        base_model: Optional[BaseModel] = None,
        limit: int = 100,
    ) -> list[ModelMetadata]:
        """
        List models with optional filters.

        Args:
            source: Filter by source
            model_type: Filter by model type
            base_model: Filter by base model
            limit: Maximum results

        Returns:
            List of models
        """
        conditions = []
        params = []

        if source:
            conditions.append("source = ?")
            params.append(source.value)

        if model_type:
            conditions.append("type = ?")
            params.append(model_type.value)

        if base_model:
            conditions.append("base_model = ?")
            params.append(base_model.value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM models WHERE {where_clause} LIMIT ?"
        params.append(limit)

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

            return [self._row_to_metadata(row) for row in rows]

    def update_model(self, metadata: ModelMetadata) -> None:
        """
        Update model metadata.

        Args:
            metadata: Updated metadata
        """
        metadata.updated_at = datetime.now()
        self.register_model(metadata)  # INSERT OR REPLACE

    def delete_model(self, model_id: str) -> bool:
        """
        Delete model from registry.

        Args:
            model_id: Model ID to delete

        Returns:
            True if deleted
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
            conn.commit()

            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted model: {model_id}")
            return deleted

    def cleanup_cache(self, keep_recent: int = 10, max_size_gb: float = 50.0) -> dict:
        """
        Clean up cached models.

        Args:
            keep_recent: Number of recent models to keep
            max_size_gb: Maximum total cache size in GB

        Returns:
            Cleanup report
        """
        # Get all models with local paths
        models = self.list_models(limit=1000)
        cached_models = [m for m in models if m.is_downloaded]

        # Sort by last access (using updated_at as proxy)
        cached_models.sort(key=lambda m: m.updated_at, reverse=True)

        # Calculate current size
        total_size = sum(m.size_gb for m in cached_models)

        deleted_count = 0
        freed_space = 0.0

        # Keep only recent models or until size is under limit
        for i, model in enumerate(cached_models):
            if i < keep_recent:
                continue

            if total_size <= max_size_gb:
                break

            # Delete model file
            if model.local_path and model.local_path.exists():
                try:
                    model.local_path.unlink()
                    freed_space += model.size_gb
                    total_size -= model.size_gb
                    deleted_count += 1

                    # Update metadata
                    model.local_path = None
                    self.update_model(model)

                    logger.info(f"Cleaned up cached model: {model.model_id}")

                except Exception as e:
                    logger.warning(f"Failed to delete {model.local_path}: {e}")

        report = {
            "deleted_count": deleted_count,
            "freed_space_gb": freed_space,
            "remaining_size_gb": total_size,
            "kept_models": cached_models[:keep_recent],
        }

        logger.info(
            f"Cache cleanup: deleted {deleted_count} models, freed {freed_space:.2f} GB"
        )

        return report

    def find_or_download(
        self,
        query: str,
        model_type: Optional[ModelType] = None,
        base_model: Optional[BaseModel] = None,
        auto_download: bool = True,
    ) -> Optional[ModelMetadata]:
        """
        Find model in registry, or search and download from hubs.

        This is the main entry point for "intelligent model acquisition".

        Workflow:
        1. Search local registry
        2. If not found, search HuggingFace
        3. If not found, search CivitAI
        4. If auto_download=True, download and register
        5. Return model metadata

        Args:
            query: Model name or ID
            model_type: Filter by model type
            base_model: Filter by base model
            auto_download: Automatically download if found remotely

        Returns:
            Model metadata or None if not found anywhere

        Example:
            >>> registry = ModelRegistry()
            >>> # This will search locally, then HF/CivitAI, then download
            >>> model = registry.find_or_download(
            ...     "stable-diffusion-xl-base-1.0",
            ...     model_type=ModelType.BASE_MODEL,
            ...     base_model=BaseModel.SDXL
            ... )
        """
        logger.info(
            f"find_or_download: query='{query}', type={model_type}, base={base_model}"
        )

        # Step 1: Search local registry
        local_results = self.search(
            query=query,
            model_type=model_type,
            base_model=base_model,
            limit=1,
        )

        if local_results and local_results[0].is_downloaded:
            logger.info(f"Found in local registry: {local_results[0].model_id}")
            return local_results[0]

        # Step 2: Search HuggingFace
        logger.info("Not found locally, searching HuggingFace...")
        try:
            hf_results = self.hf_service.search_models(
                query=query,
                model_type=model_type,
                limit=3,
            )

            # Filter by base_model if specified
            if base_model and hf_results:
                hf_results = [m for m in hf_results if m.base_model == base_model]

            if hf_results:
                best_match = hf_results[0]  # Take highest ranked
                logger.info(f"Found on HuggingFace: {best_match.model_id}")

                if auto_download:
                    logger.info(f"Downloading from HuggingFace: {best_match.model_id}")
                    result = self.hf_service.download_model(best_match.model_id)

                    if result.success:
                        best_match.local_path = result.local_path
                        best_match.sha256 = result.actual_sha256
                        self.register_model(best_match)
                        logger.info(
                            f"✅ Downloaded and registered: {best_match.model_id}"
                        )
                        return best_match
                    else:
                        logger.warning(f"Download failed: {result.error_message}")
                else:
                    # Register without download
                    self.register_model(best_match)
                    return best_match

        except Exception as e:
            logger.warning(f"HuggingFace search failed: {e}")

        # Step 3: Search CivitAI
        logger.info("Not found on HuggingFace, searching CivitAI...")
        try:
            civitai_results = self.civitai_service.search_models(
                query=query,
                model_types=[model_type] if model_type else None,
                base_model=base_model,
                limit=3,
            )

            if civitai_results:
                best_match = max(civitai_results, key=lambda m: m.download_count)
                logger.info(f"Found on CivitAI: {best_match.model_id}")

                if auto_download:
                    logger.info(f"Downloading from CivitAI: {best_match.model_id}")
                    # Extract numeric ID from model_id (format: "civitai:12345")
                    civitai_id = int(best_match.model_id.split(":")[-1])
                    result = self.civitai_service.download_model(civitai_id)

                    if result.success:
                        best_match.local_path = result.local_path
                        best_match.sha256 = result.actual_sha256
                        self.register_model(best_match)
                        logger.info(
                            f"✅ Downloaded and registered: {best_match.model_id}"
                        )
                        return best_match
                    else:
                        logger.warning(f"Download failed: {result.error_message}")
                else:
                    # Register without download
                    self.register_model(best_match)
                    return best_match

        except Exception as e:
            logger.warning(f"CivitAI search failed: {e}")

        # Not found anywhere
        logger.warning(f"Model not found anywhere: {query}")
        return None

    def ensure_downloaded(self, model_id: str) -> bool:
        """
        Ensure model is downloaded locally.

        If model is in registry but not downloaded, download it.

        Args:
            model_id: Model ID

        Returns:
            True if model is available locally

        Example:
            >>> registry.ensure_downloaded("stabilityai/sdxl-base-1.0")
        """
        model = self.get_model(model_id)

        if not model:
            logger.warning(f"Model not in registry: {model_id}")
            return False

        if model.is_downloaded:
            logger.info(f"Model already downloaded: {model_id}")
            return True

        # Download based on source
        logger.info(f"Downloading model: {model_id}")

        try:
            if model.source == Source.HUGGINGFACE:
                result = self.hf_service.download_model(model_id)
            elif model.source == Source.CIVITAI:
                civitai_id = int(model_id.split(":")[-1])
                result = self.civitai_service.download_model(civitai_id)
            else:
                logger.error(f"Unknown source: {model.source}")
                return False

            if result.success:
                model.local_path = result.local_path
                model.sha256 = result.actual_sha256
                self.update_model(model)
                logger.info(f"✅ Downloaded: {model_id}")
                return True
            else:
                logger.error(f"Download failed: {result.error_message}")
                return False

        except Exception as e:
            logger.error(f"Download error: {e}")
            return False

    def get_stats(self) -> dict:
        """
        Get registry statistics.

        Returns:
            Dictionary with registry stats
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            # Total models
            total = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]

            # By source
            by_source = {}
            cursor = conn.execute("SELECT source, COUNT(*) FROM models GROUP BY source")
            for row in cursor:
                by_source[row[0]] = row[1]

            # By type
            by_type = {}
            cursor = conn.execute("SELECT type, COUNT(*) FROM models GROUP BY type")
            for row in cursor:
                by_type[row[0]] = row[1]

            # Downloaded
            downloaded = conn.execute(
                "SELECT COUNT(*) FROM models WHERE local_path IS NOT NULL"
            ).fetchone()[0]

            # Total cache size
            cache_size_bytes = (
                conn.execute(
                    "SELECT SUM(size_bytes) FROM models WHERE local_path IS NOT NULL"
                ).fetchone()[0]
                or 0
            )

            return {
                "total_models": total,
                "by_source": by_source,
                "by_type": by_type,
                "downloaded": downloaded,
                "cache_size_gb": cache_size_bytes / (1024**3),
            }

    def _row_to_metadata(self, row: sqlite3.Row) -> ModelMetadata:
        """Convert database row to ModelMetadata."""
        return ModelMetadata(
            model_id=row["model_id"],
            name=row["name"],
            source=Source(row["source"]),
            type=ModelType(row["type"]),
            base_model=BaseModel(row["base_model"]),
            version=row["version"],
            format=row["format"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"] or "",
            trigger_words=json.loads(row["trigger_words"] or "[]"),
            tags=json.loads(row["tags"] or "[]"),
            description=row["description"] or "",
            download_count=row["download_count"] or 0,
            rating=row["rating"] or 0.0,
            recommended_weight=row["recommended_weight"],
            local_path=Path(row["local_path"]) if row["local_path"] else None,
            remote_url=row["remote_url"] or "",
            created_at=datetime.fromisoformat(row["created_at"])
            if row["created_at"]
            else datetime.now(),
            updated_at=datetime.fromisoformat(row["updated_at"])
            if row["updated_at"]
            else datetime.now(),
        )
