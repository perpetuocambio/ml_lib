"""Tests for LoRA domain entity."""

import pytest
from pathlib import Path
from ml_lib.diffusion.domain.entities.lora import LoRA, LoRARecommendation
from ml_lib.diffusion.domain.value_objects.weights import LoRAWeight, ConfidenceScore


class TestLoRAEntity:
    """Tests for LoRA entity."""

    @pytest.fixture
    def sample_lora_path(self, tmp_path):
        """Create a temporary LoRA file for testing."""
        lora_file = tmp_path / "test_lora.safetensors"
        lora_file.write_text("fake lora content")
        return lora_file

    @pytest.fixture
    def sample_lora(self, sample_lora_path):
        """Create a sample LoRA for testing."""
        return LoRA.create(
            name="anime_style",
            path=sample_lora_path,
            base_model="SDXL",
            weight=1.0,
            trigger_words=["anime", "manga style"],
            tags=["anime", "illustration", "2d"],
            download_count=10000,
            rating=4.5,
        )

    def test_create_valid_lora(self, sample_lora_path):
        """Test creating valid LoRA."""
        lora = LoRA.create(
            name="test_lora",
            path=sample_lora_path,
            base_model="SDXL",
        )
        assert lora.name == "test_lora"
        assert lora.base_model == "SDXL"
        assert lora.weight.value == 1.0  # Default

    def test_reject_empty_name(self, sample_lora_path):
        """Test rejecting LoRA with empty name."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            LoRA.create(
                name="",
                path=sample_lora_path,
                base_model="SDXL",
            )

    def test_reject_nonexistent_file(self, tmp_path):
        """Test rejecting LoRA with non-existent file."""
        fake_path = tmp_path / "nonexistent.safetensors"
        with pytest.raises(ValueError, match="file not found"):
            LoRA.create(
                name="test",
                path=fake_path,
                base_model="SDXL",
            )

    def test_reject_empty_base_model(self, sample_lora_path):
        """Test rejecting LoRA without base model."""
        with pytest.raises(ValueError, match="must specify base_model"):
            LoRA.create(
                name="test",
                path=sample_lora_path,
                base_model="",
            )

    def test_matches_prompt_with_trigger_word(self, sample_lora):
        """Test matching prompt with trigger word."""
        assert sample_lora.matches_prompt("a beautiful anime girl")
        assert sample_lora.matches_prompt("manga style character")
        assert not sample_lora.matches_prompt("photorealistic portrait")

    def test_matches_prompt_with_tag(self, sample_lora):
        """Test matching prompt with tag."""
        assert sample_lora.matches_prompt("2d illustration of a cat")
        assert not sample_lora.matches_prompt("3d render of a building")

    def test_calculate_relevance_high(self, sample_lora):
        """Test calculating high relevance score."""
        # Multiple trigger words
        score = sample_lora.calculate_relevance("anime manga style illustration")
        assert score.value > 0.5  # Should be high

    def test_calculate_relevance_low(self, sample_lora):
        """Test calculating low relevance score."""
        # No matches
        score = sample_lora.calculate_relevance("realistic photo of a car")
        assert score.value < 0.3  # Should be low (only popularity bonus)

    def test_calculate_relevance_medium(self, sample_lora):
        """Test calculating medium relevance score."""
        # One tag match
        score = sample_lora.calculate_relevance("2d artwork")
        assert 0.2 < score.value < 0.6

    def test_is_compatible_exact_match(self, sample_lora):
        """Test compatibility with exact base model match."""
        assert sample_lora.is_compatible_with("SDXL")

    def test_is_compatible_sdxl_variants(self, sample_lora):
        """Test SDXL variant compatibility."""
        assert sample_lora.is_compatible_with("SDXL 1.0")
        assert sample_lora.is_compatible_with("sdxl_base")

    def test_is_incompatible_different_model(self, sample_lora):
        """Test incompatibility with different model."""
        assert not sample_lora.is_compatible_with("SD 1.5")
        assert not sample_lora.is_compatible_with("Flux")

    def test_is_compatible_pony_with_sdxl(self, sample_lora_path):
        """Test Pony LoRA compatible with SDXL."""
        pony_lora = LoRA.create(
            name="pony_style",
            path=sample_lora_path,
            base_model="Pony",
        )
        assert pony_lora.is_compatible_with("SDXL")
        assert pony_lora.is_compatible_with("Pony")

    def test_scale_weight(self, sample_lora):
        """Test scaling LoRA weight."""
        scaled = sample_lora.scale_weight(0.8)
        assert scaled.weight.value == 0.8
        assert scaled.name == sample_lora.name  # Other props unchanged
        assert scaled is not sample_lora  # New instance

    def test_get_popularity_score(self, sample_lora):
        """Test popularity score calculation."""
        score = sample_lora.get_popularity_score()
        assert 0 <= score <= 100
        assert score > 50  # Should be reasonably high given downloads + rating

    def test_get_popularity_score_low(self, sample_lora_path):
        """Test popularity score for unpopular LoRA."""
        unpopular = LoRA.create(
            name="unpopular",
            path=sample_lora_path,
            base_model="SDXL",
            download_count=10,
            rating=2.0,
        )
        score = unpopular.get_popularity_score()
        assert score < 50

    def test_string_representation(self, sample_lora):
        """Test string representation."""
        s = str(sample_lora)
        assert "anime_style" in s
        assert "SDXL" in s

    def test_repr(self, sample_lora):
        """Test debug representation."""
        r = repr(sample_lora)
        assert "anime_style" in r
        assert "triggers=2" in r


class TestLoRARecommendation:
    """Tests for LoRARecommendation."""

    @pytest.fixture
    def sample_lora(self, tmp_path):
        """Create a sample LoRA."""
        lora_file = tmp_path / "test.safetensors"
        lora_file.write_text("fake")
        return LoRA.create(
            name="test_lora",
            path=lora_file,
            base_model="SDXL",
            trigger_words=["anime"],
            tags=["illustration"],
        )

    def test_create_recommendation(self, sample_lora):
        """Test creating recommendation."""
        rec = LoRARecommendation.create(
            lora=sample_lora,
            prompt="anime girl",
        )
        assert rec.lora == sample_lora
        assert rec.confidence.value > 0
        assert rec.reasoning  # Should have reasoning

    def test_recommendation_auto_reasoning_trigger(self, sample_lora):
        """Test automatic reasoning generation for trigger word."""
        rec = LoRARecommendation.create(
            lora=sample_lora,
            prompt="anime character",
        )
        assert "trigger" in rec.reasoning.lower()

    def test_recommendation_auto_reasoning_tag(self, sample_lora):
        """Test automatic reasoning for tag match."""
        rec = LoRARecommendation.create(
            lora=sample_lora,
            prompt="illustration of a cat",
        )
        assert "tag" in rec.reasoning.lower() or "relevant" in rec.reasoning.lower()

    def test_recommendation_custom_reasoning(self, sample_lora):
        """Test custom reasoning."""
        rec = LoRARecommendation.create(
            lora=sample_lora,
            prompt="test",
            reasoning="Custom reason here",
        )
        assert rec.reasoning == "Custom reason here"

    def test_is_confident_high(self, sample_lora):
        """Test is_confident with high score."""
        rec = LoRARecommendation(
            lora=sample_lora,
            confidence=ConfidenceScore(0.85),
            reasoning="test",
        )
        assert rec.is_confident()

    def test_is_confident_low(self, sample_lora):
        """Test is_confident with low score."""
        rec = LoRARecommendation(
            lora=sample_lora,
            confidence=ConfidenceScore(0.3),
            reasoning="test",
        )
        assert not rec.is_confident()

    def test_reject_empty_reasoning(self, sample_lora):
        """Test rejecting recommendation without reasoning."""
        with pytest.raises(ValueError, match="must have reasoning"):
            LoRARecommendation(
                lora=sample_lora,
                confidence=ConfidenceScore(0.5),
                reasoning="",
            )

    def test_string_representation(self, sample_lora):
        """Test string representation."""
        rec = LoRARecommendation.create(
            lora=sample_lora,
            prompt="test",
        )
        s = str(rec)
        assert "test_lora" in s
        assert "confidence" in s.lower()
