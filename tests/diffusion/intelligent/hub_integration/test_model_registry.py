"""Tests for ModelRegistry."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from ml_lib.diffusion.intelligent.hub_integration.model_registry import ModelRegistry
from ml_lib.diffusion.intelligent.hub_integration.entities import (
    ModelMetadata,
    Source,
    ModelType,
    BaseModel,
    ModelFormat,
)


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    yield db_path

    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def sample_metadata():
    """Create sample model metadata."""
    return ModelMetadata(
        model_id="test_model_1",
        name="Test Model",
        source=Source.HUGGINGFACE,
        type=ModelType.LORA,
        base_model=BaseModel.SDXL,
        version="1.0",
        format=ModelFormat.SAFETENSORS,
        size_bytes=1024 * 1024 * 100,  # 100 MB
        trigger_words=["anime", "style"],
        tags=["art", "anime"],
        description="Test model for unit tests",
        download_count=1000,
        rating=4.5,
    )


def test_registry_initialization(temp_db):
    """Test registry initialization."""
    registry = ModelRegistry(db_path=temp_db)

    assert registry.db_path == temp_db
    assert temp_db.exists()


def test_register_and_get_model(temp_db, sample_metadata):
    """Test registering and retrieving a model."""
    registry = ModelRegistry(db_path=temp_db)

    # Register model
    registry.register_model(sample_metadata)

    # Retrieve model
    retrieved = registry.get_model(sample_metadata.model_id)

    assert retrieved is not None
    assert retrieved.model_id == sample_metadata.model_id
    assert retrieved.name == sample_metadata.name
    assert retrieved.source == sample_metadata.source
    assert retrieved.type == sample_metadata.type
    assert retrieved.trigger_words == sample_metadata.trigger_words


def test_list_models_with_filters(temp_db):
    """Test listing models with filters."""
    registry = ModelRegistry(db_path=temp_db)

    # Register multiple models
    models = [
        ModelMetadata(
            model_id=f"model_{i}",
            name=f"Model {i}",
            source=Source.HUGGINGFACE if i % 2 == 0 else Source.CIVITAI,
            type=ModelType.LORA if i % 3 == 0 else ModelType.BASE_MODEL,
            base_model=BaseModel.SDXL,
        )
        for i in range(10)
    ]

    for model in models:
        registry.register_model(model)

    # Test filtering by source
    hf_models = registry.list_models(source=Source.HUGGINGFACE)
    assert len(hf_models) == 5

    # Test filtering by type
    lora_models = registry.list_models(model_type=ModelType.LORA)
    assert len(lora_models) == 4  # 0, 3, 6, 9


def test_search_models(temp_db, sample_metadata):
    """Test search functionality."""
    registry = ModelRegistry(db_path=temp_db)

    registry.register_model(sample_metadata)

    # Search by name
    results = registry.search(query="Test")
    assert len(results) == 1
    assert results[0].model_id == sample_metadata.model_id

    # Search by tag
    results = registry.search(query="anime")
    assert len(results) == 1


def test_update_model(temp_db, sample_metadata):
    """Test updating model metadata."""
    registry = ModelRegistry(db_path=temp_db)

    # Register original
    registry.register_model(sample_metadata)

    # Update
    sample_metadata.rating = 5.0
    registry.update_model(sample_metadata)

    # Verify update
    updated = registry.get_model(sample_metadata.model_id)
    assert updated.rating == 5.0


def test_delete_model(temp_db, sample_metadata):
    """Test deleting a model."""
    registry = ModelRegistry(db_path=temp_db)

    # Register and delete
    registry.register_model(sample_metadata)
    result = registry.delete_model(sample_metadata.model_id)

    assert result is True

    # Verify deletion
    deleted = registry.get_model(sample_metadata.model_id)
    assert deleted is None


def test_cleanup_cache(temp_db):
    """Test cache cleanup."""
    registry = ModelRegistry(db_path=temp_db)

    # Create temporary files
    temp_dir = Path(tempfile.gettempdir()) / "test_models"
    temp_dir.mkdir(exist_ok=True)

    # Register models with local paths
    for i in range(15):
        model_path = temp_dir / f"model_{i}.safetensors"
        model_path.write_bytes(b"dummy" * 1024 * 200)  # ~1 MB each

        metadata = ModelMetadata(
            model_id=f"model_{i}",
            name=f"Model {i}",
            source=Source.HUGGINGFACE,
            type=ModelType.LORA,
            base_model=BaseModel.SDXL,
            size_bytes=model_path.stat().st_size,
            local_path=model_path,
        )
        registry.register_model(metadata)

    # Cleanup keeping only 5 recent
    report = registry.cleanup_cache(keep_recent=5, max_size_gb=0.001)

    assert report["deleted_count"] > 0
    assert len(report["kept_models"]) == 5

    # Cleanup temp
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_model_metadata_serialization(sample_metadata):
    """Test metadata serialization to/from dict."""
    # To dict
    data = sample_metadata.to_dict()

    assert data["model_id"] == sample_metadata.model_id
    assert data["trigger_words"] == sample_metadata.trigger_words

    # From dict
    restored = ModelMetadata.from_dict(data)

    assert restored.model_id == sample_metadata.model_id
    assert restored.trigger_words == sample_metadata.trigger_words
    assert restored.source == sample_metadata.source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
