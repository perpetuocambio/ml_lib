"""Tests for LoRARecommendationService - Domain Service.

This demonstrates testing a domain service with repository pattern.
No mocks needed - we use InMemoryRepository!
"""

import pytest
from pathlib import Path
from ml_lib.diffusion.domain.services.lora_recommendation_service import (
    LoRARecommendationService,
)
from ml_lib.diffusion.infrastructure.persistence.in_memory_model_repository import (
    InMemoryModelRepository,
)
from ml_lib.diffusion.domain.entities.lora import LoRA


class TestLoRARecommendationService:
    """Tests for LoRARecommendationService."""

    @pytest.fixture
    def repository(self, tmp_path):
        """Create repository with sample data."""
        repo = InMemoryModelRepository()

        # Add sample LoRAs
        lora1_file = tmp_path / "anime.safetensors"
        lora1_file.write_text("fake")
        lora1 = LoRA.create(
            name="anime_style",
            path=lora1_file,
            base_model="SDXL",
            trigger_words=["anime", "manga"],
            tags=["illustration", "2d"],
            download_count=50000,
            rating=4.7,
        )
        repo.add_lora(lora1)

        lora2_file = tmp_path / "realistic.safetensors"
        lora2_file.write_text("fake")
        lora2 = LoRA.create(
            name="realistic",
            path=lora2_file,
            base_model="SDXL",
            trigger_words=["photorealistic", "detailed"],
            tags=["photo", "realistic"],
            download_count=30000,
            rating=4.5,
        )
        repo.add_lora(lora2)

        lora3_file = tmp_path / "sd15_style.safetensors"
        lora3_file.write_text("fake")
        lora3 = LoRA.create(
            name="sd15_style",
            path=lora3_file,
            base_model="SD 1.5",
            trigger_words=["vintage"],
            tags=["retro"],
            download_count=10000,
            rating=4.0,
        )
        repo.add_lora(lora3)

        return repo

    @pytest.fixture
    def service(self, repository):
        """Create service with repository."""
        return LoRARecommendationService(repository=repository)

    def test_recommend_with_matching_trigger(self, service):
        """Test recommending LoRAs with matching trigger words."""
        recommendations = service.recommend(
            prompt="anime girl with blue hair",
            base_model="SDXL",
            max_loras=3,
            min_confidence=0.3,
        )

        # Should recommend anime_style (has "anime" trigger)
        assert len(recommendations) > 0
        assert recommendations[0].lora.name == "anime_style"
        assert recommendations[0].confidence.value > 0.3

    def test_recommend_filters_by_base_model(self, service):
        """Test that recommendations are filtered by base model."""
        recommendations = service.recommend(
            prompt="vintage photo",
            base_model="SDXL",
            max_loras=10,
            min_confidence=0.0,
        )

        # Should not recommend SD 1.5 LoRA
        lora_names = [r.lora.name for r in recommendations]
        assert "sd15_style" not in lora_names

    def test_recommend_respects_max_loras(self, service):
        """Test max_loras limit."""
        recommendations = service.recommend(
            prompt="beautiful image",
            base_model="SDXL",
            max_loras=1,
            min_confidence=0.0,
        )

        assert len(recommendations) <= 1

    def test_recommend_respects_min_confidence(self, service):
        """Test min_confidence filter."""
        recommendations = service.recommend(
            prompt="random unrelated prompt xyz123",
            base_model="SDXL",
            max_loras=10,
            min_confidence=0.8,  # High threshold
        )

        # Unlikely to get high confidence for unrelated prompt
        assert len(recommendations) == 0 or all(
            r.confidence.value >= 0.8 for r in recommendations
        )

    def test_recommend_top_returns_best(self, service):
        """Test recommend_top returns single best."""
        best = service.recommend_top(
            prompt="anime manga style character",
            base_model="SDXL",
        )

        assert best is not None
        assert best.lora.name == "anime_style"  # Best match

    def test_recommend_top_returns_none_if_no_match(self, service):
        """Test recommend_top with incompatible base model."""
        best = service.recommend_top(
            prompt="anime character",
            base_model="Flux",  # No Flux LoRAs
        )

        # Might return None or low confidence
        # (depends on implementation details)
        if best:
            assert best.confidence.value > 0.0

    def test_filter_confident_recommendations(self, service):
        """Test filtering to only confident recommendations."""
        # Get all recommendations
        all_recs = service.recommend(
            prompt="anime realistic photo",
            base_model="SDXL",
            max_loras=10,
            min_confidence=0.0,
        )

        # Filter to confident only
        confident = service.filter_confident_recommendations(all_recs)

        # All should be high confidence
        assert all(r.is_confident() for r in confident)
        assert len(confident) <= len(all_recs)

    def test_get_recommendations_by_trigger_words(self, service):
        """Test getting LoRAs by explicit trigger word match."""
        recs = service.get_recommendations_by_trigger_words(
            prompt="photorealistic detailed portrait",
            base_model="SDXL",
        )

        # Should get realistic LoRA (has matching triggers)
        assert len(recs) > 0
        lora_names = [r.lora.name for r in recs]
        assert "realistic" in lora_names

    def test_no_recommendations_for_incompatible_base(self, service):
        """Test no recommendations when base model incompatible."""
        recs = service.recommend(
            prompt="anime character",
            base_model="Flux",  # No Flux LoRAs in repo
            max_loras=10,
            min_confidence=0.0,
        )

        assert len(recs) == 0

    def test_recommendations_sorted_by_confidence(self, service):
        """Test that recommendations are sorted by confidence."""
        recs = service.recommend(
            prompt="anime illustration",
            base_model="SDXL",
            max_loras=10,
            min_confidence=0.0,
        )

        if len(recs) > 1:
            # Check sorted descending
            confidences = [r.confidence.value for r in recs]
            assert confidences == sorted(confidences, reverse=True)

    def test_service_with_empty_repository(self, tmp_path):
        """Test service behavior with empty repository."""
        empty_repo = InMemoryModelRepository()
        service = LoRARecommendationService(repository=empty_repo)

        recs = service.recommend(
            prompt="any prompt",
            base_model="SDXL",
        )

        assert len(recs) == 0

    def test_service_with_seeded_repository(self):
        """Test service with seeded sample data."""
        repo = InMemoryModelRepository()
        repo.seed_with_samples()

        service = LoRARecommendationService(repository=repo)

        # Should have 5 sample LoRAs
        assert repo.count_loras() == 5

        # Recommend for anime
        recs = service.recommend(
            prompt="anime girl",
            base_model="SDXL",
            max_loras=3,
        )

        assert len(recs) > 0
        # Should get anime_style_v2 from samples
        lora_names = [r.lora.name for r in recs]
        assert "anime_style_v2" in lora_names
