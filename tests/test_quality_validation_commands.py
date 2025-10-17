"""Tests for Quality Validation Commands and Domain Service.

Tests cover:
- QualityValidationService domain logic
- ValidateImageQualityCommand and handler
- ValidateBatchQualityCommand and handler
- Integration with CommandBus
- Quality metrics calculation
- Error handling
"""

import pytest
from pathlib import Path
from PIL import Image
import tempfile
from ml_lib.diffusion.application.commands import (
    CommandBus,
    ValidateImageQualityCommand,
    ValidateBatchQualityCommand,
    ValidateImageQualityHandler,
    ValidateBatchQualityHandler,
)
from ml_lib.diffusion.domain.services.quality_validation_service import (
    QualityValidationService,
    QualityMetrics,
)


@pytest.fixture
def validation_service():
    """Create validation service."""
    return QualityValidationService(min_quality_threshold=70.0)


@pytest.fixture
def command_bus(validation_service):
    """Create command bus with quality validation handlers."""
    bus = CommandBus()
    bus.register(ValidateImageQualityCommand, ValidateImageQualityHandler(validation_service))
    bus.register(ValidateBatchQualityCommand, ValidateBatchQualityHandler(validation_service))
    return bus


@pytest.fixture
def good_image():
    """Create a good quality test image."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (1024, 1024), color=(128, 128, 128))
        img.save(f.name)
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def bad_resolution_image():
    """Create image with wrong resolution."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (512, 512), color=(128, 128, 128))
        img.save(f.name)
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def dark_image():
    """Create very dark image (possible artifact)."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (1024, 1024), color=(2, 2, 2))
        img.save(f.name)
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


# ============================================================================
# Domain Service Tests
# ============================================================================

def test_quality_validation_service_good_image(validation_service, good_image):
    """Test validation of good quality image."""
    metrics = validation_service.validate_image(good_image)

    assert metrics.exists is True
    assert metrics.valid_format is True
    assert metrics.correct_resolution is True
    assert metrics.no_corruption is True
    assert metrics.quality_score >= 70.0
    assert metrics.is_valid is True
    assert metrics.actual_width == 1024
    assert metrics.actual_height == 1024
    assert len(metrics.errors) == 0


def test_quality_validation_service_wrong_resolution(validation_service, bad_resolution_image):
    """Test validation of image with wrong resolution."""
    metrics = validation_service.validate_image(bad_resolution_image)

    assert metrics.exists is True
    assert metrics.valid_format is True
    assert metrics.correct_resolution is False
    assert metrics.actual_width == 512
    assert metrics.actual_height == 512
    assert metrics.quality_score < 100.0


def test_quality_validation_service_dark_image(validation_service, dark_image):
    """Test validation of very dark image."""
    metrics = validation_service.validate_image(dark_image)

    assert metrics.exists is True
    assert metrics.valid_format is True
    # Very dark images should have low color range scores
    assert metrics.valid_color_range is False


def test_quality_validation_service_nonexistent_file(validation_service):
    """Test validation of non-existent file."""
    metrics = validation_service.validate_image(Path("/tmp/does_not_exist.png"))

    assert metrics.exists is False
    assert metrics.quality_score == 0.0
    assert metrics.is_valid is False
    assert len(metrics.errors) > 0
    assert "does not exist" in metrics.errors[0]


def test_quality_metrics_properties(validation_service, good_image):
    """Test QualityMetrics computed properties."""
    metrics = validation_service.validate_image(good_image)

    # Test is_valid property (>= 70)
    if metrics.quality_score >= 70.0:
        assert metrics.is_valid is True
    else:
        assert metrics.is_valid is False

    # Test is_excellent property (>= 90)
    if metrics.quality_score >= 90.0:
        assert metrics.is_excellent is True
    else:
        assert metrics.is_excellent is False

    # Test resolution_matches property
    if metrics.actual_width == 1024 and metrics.actual_height == 1024:
        assert metrics.resolution_matches is True
    else:
        assert metrics.resolution_matches is False


def test_quality_validation_batch(validation_service, good_image, bad_resolution_image):
    """Test batch validation."""
    metrics_list = validation_service.validate_batch([good_image, bad_resolution_image])

    assert len(metrics_list) == 2
    assert all(isinstance(m, QualityMetrics) for m in metrics_list)

    # First should be good
    assert metrics_list[0].quality_score >= 70.0
    # Second has wrong resolution
    assert metrics_list[1].correct_resolution is False


def test_quality_validation_summary_stats(validation_service, good_image, bad_resolution_image, dark_image):
    """Test summary statistics calculation."""
    metrics_list = validation_service.validate_batch([good_image, bad_resolution_image, dark_image])
    summary = validation_service.get_summary_stats(metrics_list)

    assert summary["total"] == 3
    assert "passed" in summary
    assert "failed" in summary
    assert "pass_rate" in summary
    assert "avg_quality" in summary
    assert "min_quality" in summary
    assert "max_quality" in summary
    assert 0.0 <= summary["pass_rate"] <= 100.0


def test_quality_validation_empty_batch(validation_service):
    """Test empty batch returns zero stats."""
    summary = validation_service.get_summary_stats([])

    assert summary["total"] == 0
    assert summary["passed"] == 0
    assert summary["failed"] == 0
    assert summary["pass_rate"] == 0.0
    assert summary["avg_quality"] == 0.0


# ============================================================================
# Command Tests
# ============================================================================

def test_validate_image_quality_command_success(command_bus, good_image):
    """Test ValidateImageQualityCommand with good image."""
    command = ValidateImageQualityCommand(
        image_path=good_image,
        expected_width=1024,
        expected_height=1024,
        min_quality_threshold=70.0,
    )

    result = command_bus.dispatch(command)

    assert result.is_success is True
    assert result.data is not None
    assert isinstance(result.data, QualityMetrics)
    assert result.metadata["quality_score"] >= 70.0
    assert result.metadata["is_valid"] is True


def test_validate_image_quality_command_below_threshold(command_bus, dark_image):
    """Test ValidateImageQualityCommand with image below threshold."""
    command = ValidateImageQualityCommand(
        image_path=dark_image,
        expected_width=1024,
        expected_height=1024,
        min_quality_threshold=90.0,  # High threshold
    )

    result = command_bus.dispatch(command)

    # Should fail if quality is below 90
    # Note: When threshold fails, data is None and error message is set
    if not result.is_success:
        assert "below threshold" in result.error
    else:
        # If it passed, quality must be >= 90
        assert result.data.quality_score >= 90.0


def test_validate_image_quality_command_nonexistent_file(command_bus):
    """Test ValidateImageQualityCommand with non-existent file."""
    command = ValidateImageQualityCommand(
        image_path=Path("/tmp/does_not_exist.png"),
    )

    result = command_bus.dispatch(command)

    assert result.is_success is False


def test_validate_image_quality_command_validation_errors(command_bus, good_image):
    """Test ValidateImageQualityCommand validation errors."""
    # Test invalid dimensions
    command = ValidateImageQualityCommand(
        image_path=good_image,
        expected_width=-1,
        expected_height=1024,
    )

    result = command_bus.dispatch(command)

    assert result.is_success is False
    assert "positive" in result.error.lower()

    # Test invalid threshold
    command = ValidateImageQualityCommand(
        image_path=good_image,
        min_quality_threshold=150.0,  # > 100
    )

    result = command_bus.dispatch(command)

    assert result.is_success is False
    assert "between 0 and 100" in result.error


def test_validate_batch_quality_command_success(command_bus, good_image, bad_resolution_image):
    """Test ValidateBatchQualityCommand with multiple images."""
    command = ValidateBatchQualityCommand(
        image_paths=[good_image, bad_resolution_image],
        expected_width=1024,
        expected_height=1024,
        min_quality_threshold=50.0,  # Lower threshold for pass rate
    )

    result = command_bus.dispatch(command)

    assert result.is_success is True
    assert isinstance(result.data, list)
    assert len(result.data) == 2
    assert all(isinstance(m, QualityMetrics) for m in result.data)
    assert result.metadata["total_images"] == 2
    assert "passed" in result.metadata
    assert "failed" in result.metadata
    assert "pass_rate" in result.metadata


def test_validate_batch_quality_command_below_threshold(command_bus, dark_image):
    """Test ValidateBatchQualityCommand with batch below threshold."""
    command = ValidateBatchQualityCommand(
        image_paths=[dark_image, dark_image],  # Two dark images
        expected_width=1024,
        expected_height=1024,
        min_quality_threshold=95.0,  # Very high threshold
    )

    result = command_bus.dispatch(command)

    # Should fail if pass rate is below 95%
    if result.metadata["pass_rate"] < 95.0:
        assert result.is_success is False
        assert "below threshold" in result.error


def test_validate_batch_quality_command_empty_list(command_bus):
    """Test ValidateBatchQualityCommand with empty list."""
    command = ValidateBatchQualityCommand(
        image_paths=[],
    )

    result = command_bus.dispatch(command)

    assert result.is_success is False
    assert "cannot be empty" in result.error


def test_validate_batch_quality_command_validation_errors(command_bus, good_image):
    """Test ValidateBatchQualityCommand validation errors."""
    # Test invalid dimensions
    command = ValidateBatchQualityCommand(
        image_paths=[good_image],
        expected_width=0,
        expected_height=1024,
    )

    result = command_bus.dispatch(command)

    assert result.is_success is False
    assert "positive" in result.error.lower()


# ============================================================================
# Integration Tests
# ============================================================================

def test_quality_validation_e2e_workflow(command_bus, good_image, bad_resolution_image, dark_image):
    """Test complete quality validation workflow."""
    # Step 1: Validate single good image
    single_cmd = ValidateImageQualityCommand(image_path=good_image)
    single_result = command_bus.dispatch(single_cmd)

    assert single_result.is_success is True
    good_quality = single_result.metadata["quality_score"]

    # Step 2: Validate batch
    batch_cmd = ValidateBatchQualityCommand(
        image_paths=[good_image, bad_resolution_image, dark_image],
        min_quality_threshold=30.0,  # Lower threshold for mixed batch
    )
    batch_result = command_bus.dispatch(batch_cmd)

    assert batch_result.is_success is True
    assert batch_result.metadata["total_images"] == 3
    assert batch_result.metadata["max_quality"] >= good_quality - 10  # Should be similar

    # Step 3: Validate that summary stats are consistent
    assert batch_result.metadata["passed"] + batch_result.metadata["failed"] == 3
    assert batch_result.metadata["pass_rate"] >= 0.0
    assert batch_result.metadata["pass_rate"] <= 100.0


def test_quality_validation_different_resolutions(command_bus, good_image):
    """Test validation with different expected resolutions."""
    # Validate with correct resolution
    cmd_correct = ValidateImageQualityCommand(
        image_path=good_image,
        expected_width=1024,
        expected_height=1024,
    )
    result_correct = command_bus.dispatch(cmd_correct)

    assert result_correct.is_success is True
    assert result_correct.data.correct_resolution is True

    # Validate with wrong expected resolution
    cmd_wrong = ValidateImageQualityCommand(
        image_path=good_image,
        expected_width=512,
        expected_height=512,
    )
    result_wrong = command_bus.dispatch(cmd_wrong)

    # Should succeed but report wrong resolution
    assert result_wrong.data.correct_resolution is False
    assert result_wrong.data.actual_width == 1024
    assert result_wrong.data.actual_height == 1024


def test_quality_validation_metadata_consistency(command_bus, good_image):
    """Test that metadata matches the actual metrics."""
    command = ValidateImageQualityCommand(image_path=good_image)
    result = command_bus.dispatch(command)

    assert result.is_success is True

    # Metadata should match data
    assert result.metadata["quality_score"] == result.data.quality_score
    assert result.metadata["is_valid"] == result.data.is_valid
    assert result.metadata["is_excellent"] == result.data.is_excellent
    assert result.metadata["resolution"] == f"{result.data.actual_width}x{result.data.actual_height}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
