"""Tests for SQLiteModelRepository - Real persistence implementation.

Tests database operations, CRUD, queries, and data integrity.
"""

import pytest
from pathlib import Path
from ml_lib.diffusion.infrastructure.persistence.sqlite_model_repository import (
    SQLiteModelRepository,
)
from ml_lib.diffusion.domain.entities.lora import LoRA


class TestSQLiteModelRepository:
    """Tests for SQLite repository implementation."""

    @pytest.fixture
    def db_path(self, tmp_path):
        """Create temporary database path."""
        return tmp_path / "test_loras.db"

    @pytest.fixture
    def repository(self, db_path):
        """Create repository with temp database."""
        return SQLiteModelRepository(db_path=db_path)

    @pytest.fixture
    def sample_lora(self, tmp_path):
        """Create sample LoRA for testing."""
        lora_file = tmp_path / "sample.safetensors"
        lora_file.write_text("fake")

        return LoRA.create(
            name="test_lora",
            path=lora_file,
            base_model="SDXL",
            weight=0.8,
            trigger_words=["anime", "manga"],
            tags=["illustration", "2d"],
            download_count=1000,
            rating=4.5,
        )

    def test_init_creates_database(self, db_path):
        """Test that initialization creates database file."""
        assert not db_path.exists()

        repo = SQLiteModelRepository(db_path=db_path)

        assert db_path.exists()
        assert repo.count_loras() == 0

    def test_add_lora(self, repository, sample_lora):
        """Test adding a LoRA."""
        repository.add_lora(sample_lora)

        assert repository.count_loras() == 1

        # Verify we can retrieve it
        retrieved = repository.get_lora_by_name("test_lora")
        assert retrieved is not None
        assert retrieved.name == "test_lora"
        assert retrieved.base_model == "SDXL"
        assert retrieved.weight.value == 0.8
        assert retrieved.download_count == 1000
        assert retrieved.rating == 4.5

    def test_add_duplicate_lora_raises_error(self, repository, sample_lora):
        """Test that adding duplicate LoRA raises error."""
        repository.add_lora(sample_lora)

        with pytest.raises(ValueError, match="already exists"):
            repository.add_lora(sample_lora)

    def test_add_lora_with_trigger_words(self, repository, sample_lora):
        """Test that trigger words are persisted."""
        repository.add_lora(sample_lora)

        retrieved = repository.get_lora_by_name("test_lora")
        assert retrieved is not None
        assert "anime" in retrieved.trigger_words
        assert "manga" in retrieved.trigger_words

    def test_add_lora_with_tags(self, repository, sample_lora):
        """Test that tags are persisted."""
        repository.add_lora(sample_lora)

        retrieved = repository.get_lora_by_name("test_lora")
        assert retrieved is not None
        assert "illustration" in retrieved.tags
        assert "2d" in retrieved.tags

    def test_get_lora_by_name_not_found(self, repository):
        """Test getting non-existent LoRA returns None."""
        result = repository.get_lora_by_name("nonexistent")
        assert result is None

    def test_get_all_loras(self, repository, tmp_path):
        """Test getting all LoRAs."""
        # Add multiple LoRAs
        for i in range(3):
            lora_file = tmp_path / f"lora{i}.safetensors"
            lora_file.write_text("fake")

            lora = LoRA.create(
                name=f"lora_{i}",
                path=lora_file,
                base_model="SDXL",
            )
            repository.add_lora(lora)

        all_loras = repository.get_all_loras()
        assert len(all_loras) == 3
        names = [l.name for l in all_loras]
        assert "lora_0" in names
        assert "lora_1" in names
        assert "lora_2" in names

    def test_get_loras_by_base_model(self, repository, tmp_path):
        """Test filtering by base model."""
        # Add SDXL LoRA
        lora_file1 = tmp_path / "sdxl.safetensors"
        lora_file1.write_text("fake")
        lora1 = LoRA.create(name="sdxl_lora", path=lora_file1, base_model="SDXL")
        repository.add_lora(lora1)

        # Add SD 1.5 LoRA
        lora_file2 = tmp_path / "sd15.safetensors"
        lora_file2.write_text("fake")
        lora2 = LoRA.create(name="sd15_lora", path=lora_file2, base_model="SD 1.5")
        repository.add_lora(lora2)

        # Filter by SDXL
        sdxl_loras = repository.get_loras_by_base_model("SDXL")
        assert len(sdxl_loras) == 1
        assert sdxl_loras[0].name == "sdxl_lora"

        # Filter by SD 1.5
        sd15_loras = repository.get_loras_by_base_model("SD 1.5")
        assert len(sd15_loras) == 1
        assert sd15_loras[0].name == "sd15_lora"

    def test_get_loras_by_tags(self, repository, tmp_path):
        """Test filtering by tags."""
        # Add LoRA with "anime" tag
        lora_file1 = tmp_path / "anime.safetensors"
        lora_file1.write_text("fake")
        lora1 = LoRA.create(
            name="anime_lora",
            path=lora_file1,
            base_model="SDXL",
            tags=["anime", "illustration"],
        )
        repository.add_lora(lora1)

        # Add LoRA with "realistic" tag
        lora_file2 = tmp_path / "realistic.safetensors"
        lora_file2.write_text("fake")
        lora2 = LoRA.create(
            name="realistic_lora",
            path=lora_file2,
            base_model="SDXL",
            tags=["realistic", "photo"],
        )
        repository.add_lora(lora2)

        # Search for anime
        anime_loras = repository.get_loras_by_tags(["anime"])
        assert len(anime_loras) == 1
        assert anime_loras[0].name == "anime_lora"

        # Search for illustration
        illustration_loras = repository.get_loras_by_tags(["illustration"])
        assert len(illustration_loras) == 1

        # Search for multiple tags (should match any)
        multiple_tags = repository.get_loras_by_tags(["anime", "photo"])
        assert len(multiple_tags) == 2

    def test_get_popular_loras(self, repository, tmp_path):
        """Test getting popular LoRAs sorted by downloads."""
        # Add LoRAs with different download counts
        for i, downloads in enumerate([100, 5000, 1000]):
            lora_file = tmp_path / f"lora{i}.safetensors"
            lora_file.write_text("fake")

            lora = LoRA.create(
                name=f"lora_{i}",
                path=lora_file,
                base_model="SDXL",
                download_count=downloads,
            )
            repository.add_lora(lora)

        popular = repository.get_popular_loras(limit=3)

        # Should be sorted by download count descending
        assert len(popular) == 3
        assert popular[0].name == "lora_1"  # 5000 downloads
        assert popular[1].name == "lora_2"  # 1000 downloads
        assert popular[2].name == "lora_0"  # 100 downloads

    def test_search_loras_by_name(self, repository, tmp_path):
        """Test searching LoRAs by name."""
        lora_file = tmp_path / "anime_style.safetensors"
        lora_file.write_text("fake")

        lora = LoRA.create(
            name="anime_style_v2",
            path=lora_file,
            base_model="SDXL",
        )
        repository.add_lora(lora)

        results = repository.search_loras(query="anime")
        assert len(results) == 1
        assert results[0].name == "anime_style_v2"

    def test_search_loras_by_tag(self, repository, tmp_path):
        """Test searching LoRAs by tag."""
        lora_file = tmp_path / "test.safetensors"
        lora_file.write_text("fake")

        lora = LoRA.create(
            name="test_lora",
            path=lora_file,
            base_model="SDXL",
            tags=["cyberpunk", "neon"],
        )
        repository.add_lora(lora)

        results = repository.search_loras(query="cyberpunk")
        assert len(results) == 1
        assert results[0].name == "test_lora"

    def test_search_loras_by_trigger_word(self, repository, tmp_path):
        """Test searching LoRAs by trigger word."""
        lora_file = tmp_path / "test.safetensors"
        lora_file.write_text("fake")

        lora = LoRA.create(
            name="test_lora",
            path=lora_file,
            base_model="SDXL",
            trigger_words=["magical girl", "sparkles"],
        )
        repository.add_lora(lora)

        results = repository.search_loras(query="magical")
        assert len(results) == 1
        assert results[0].name == "test_lora"

    def test_search_loras_with_base_model_filter(self, repository, tmp_path):
        """Test search with base model filter."""
        # Add SDXL LoRA
        lora_file1 = tmp_path / "sdxl.safetensors"
        lora_file1.write_text("fake")
        lora1 = LoRA.create(
            name="anime_sdxl",
            path=lora_file1,
            base_model="SDXL",
            tags=["anime"],
        )
        repository.add_lora(lora1)

        # Add SD 1.5 LoRA
        lora_file2 = tmp_path / "sd15.safetensors"
        lora_file2.write_text("fake")
        lora2 = LoRA.create(
            name="anime_sd15",
            path=lora_file2,
            base_model="SD 1.5",
            tags=["anime"],
        )
        repository.add_lora(lora2)

        # Search with SDXL filter
        results = repository.search_loras(query="anime", base_model="SDXL")
        assert len(results) == 1
        assert results[0].name == "anime_sdxl"

    def test_search_loras_with_min_rating(self, repository, tmp_path):
        """Test search with minimum rating filter."""
        # Add high-rated LoRA
        lora_file1 = tmp_path / "high.safetensors"
        lora_file1.write_text("fake")
        lora1 = LoRA.create(
            name="high_rated",
            path=lora_file1,
            base_model="SDXL",
            tags=["test"],
            rating=4.5,
        )
        repository.add_lora(lora1)

        # Add low-rated LoRA
        lora_file2 = tmp_path / "low.safetensors"
        lora_file2.write_text("fake")
        lora2 = LoRA.create(
            name="low_rated",
            path=lora_file2,
            base_model="SDXL",
            tags=["test"],
            rating=2.0,
        )
        repository.add_lora(lora2)

        # Search with min_rating filter
        results = repository.search_loras(query="test", min_rating=4.0)
        assert len(results) == 1
        assert results[0].name == "high_rated"

    def test_update_lora(self, repository, sample_lora, tmp_path):
        """Test updating existing LoRA."""
        repository.add_lora(sample_lora)

        # Modify the LoRA
        updated_lora = LoRA.create(
            name="test_lora",  # Same name
            path=sample_lora.path,
            base_model="SDXL",
            weight=1.2,  # Changed
            trigger_words=["new_trigger"],  # Changed
            tags=["new_tag"],  # Changed
            download_count=2000,  # Changed
            rating=5.0,  # Changed
        )

        repository.update_lora(updated_lora)

        # Verify changes
        retrieved = repository.get_lora_by_name("test_lora")
        assert retrieved is not None
        assert retrieved.weight.value == 1.2
        assert retrieved.download_count == 2000
        assert retrieved.rating == 5.0
        assert "new_trigger" in retrieved.trigger_words
        assert "new_tag" in retrieved.tags

    def test_update_nonexistent_lora_raises_error(self, repository, sample_lora):
        """Test updating non-existent LoRA raises error."""
        with pytest.raises(ValueError, match="not found"):
            repository.update_lora(sample_lora)

    def test_delete_lora(self, repository, sample_lora):
        """Test deleting LoRA."""
        repository.add_lora(sample_lora)
        assert repository.count_loras() == 1

        result = repository.delete_lora("test_lora")

        assert result is True
        assert repository.count_loras() == 0
        assert repository.get_lora_by_name("test_lora") is None

    def test_delete_nonexistent_lora_returns_false(self, repository):
        """Test deleting non-existent LoRA returns False."""
        result = repository.delete_lora("nonexistent")
        assert result is False

    def test_delete_cascades_to_related_data(self, repository, sample_lora):
        """Test that deleting LoRA also deletes trigger words and tags."""
        repository.add_lora(sample_lora)

        repository.delete_lora("test_lora")

        # Try to add again - should work (no orphaned data)
        repository.add_lora(sample_lora)
        assert repository.count_loras() == 1

    def test_count_loras(self, repository, tmp_path):
        """Test counting LoRAs."""
        assert repository.count_loras() == 0

        for i in range(5):
            lora_file = tmp_path / f"lora{i}.safetensors"
            lora_file.write_text("fake")

            lora = LoRA.create(
                name=f"lora_{i}",
                path=lora_file,
                base_model="SDXL",
            )
            repository.add_lora(lora)

        assert repository.count_loras() == 5

    def test_persistence_across_instances(self, db_path, tmp_path):
        """Test that data persists across repository instances."""
        lora_file = tmp_path / "persistent.safetensors"
        lora_file.write_text("fake")

        # Create first instance and add data
        repo1 = SQLiteModelRepository(db_path=db_path)
        lora = LoRA.create(
            name="persistent_lora",
            path=lora_file,
            base_model="SDXL",
        )
        repo1.add_lora(lora)

        # Create second instance - should see same data
        repo2 = SQLiteModelRepository(db_path=db_path)
        assert repo2.count_loras() == 1

        retrieved = repo2.get_lora_by_name("persistent_lora")
        assert retrieved is not None
        assert retrieved.name == "persistent_lora"
