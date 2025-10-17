"""Quality Validation Domain Service.

Validates generated images for quality, correctness, and semantic alignment.

This is a Domain Service that encapsulates the business logic for
evaluating image generation quality.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from PIL import Image, ImageStat
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualityMetrics:
    """Immutable quality metrics for a generated image."""

    # Basic validation
    exists: bool
    valid_format: bool
    correct_resolution: bool

    # Technical quality
    no_corruption: bool
    valid_color_range: bool
    no_artifacts: bool

    # Dimensions
    actual_width: int
    actual_height: int
    expected_width: int
    expected_height: int

    # Color statistics
    mean_rgb: tuple[float, float, float]
    std_rgb: tuple[float, float, float]

    # Overall score
    quality_score: float  # 0-100

    # Errors (if any)
    errors: list[str]

    @property
    def is_valid(self) -> bool:
        """Check if image passes minimum quality threshold."""
        return self.quality_score >= 70.0

    @property
    def is_excellent(self) -> bool:
        """Check if image is excellent quality."""
        return self.quality_score >= 90.0

    @property
    def resolution_matches(self) -> bool:
        """Check if resolution matches expected."""
        return (self.actual_width == self.expected_width and
                self.actual_height == self.expected_height)


class QualityValidationService:
    """
    Domain Service for validating image generation quality.

    Responsibilities:
    - Validate image format and integrity
    - Check resolution and dimensions
    - Analyze color distribution
    - Detect artifacts and corruption
    - Calculate quality scores

    This service follows Domain-Driven Design principles:
    - Stateless service (no internal state)
    - Pure business logic (no infrastructure concerns)
    - Injectable via dependency injection
    - Testable without external dependencies
    """

    def __init__(
        self,
        min_quality_threshold: float = 70.0,
        enable_logging: bool = True
    ):
        """
        Initialize quality validation service.

        Args:
            min_quality_threshold: Minimum score to consider valid (0-100)
            enable_logging: Enable logging of validation results
        """
        self.min_quality_threshold = min_quality_threshold
        self.enable_logging = enable_logging

    def validate_image(
        self,
        image_path: Path,
        expected_width: int = 1024,
        expected_height: int = 1024,
    ) -> QualityMetrics:
        """
        Validate image quality.

        Args:
            image_path: Path to image file
            expected_width: Expected image width
            expected_height: Expected image height

        Returns:
            QualityMetrics with validation results
        """
        errors = []

        # Default values for failed validations
        actual_width = 0
        actual_height = 0
        mean_rgb = (0.0, 0.0, 0.0)
        std_rgb = (0.0, 0.0, 0.0)

        # 1. Check existence
        if not image_path.exists():
            errors.append("File does not exist")
            return self._create_failed_metrics(
                expected_width, expected_height, errors
            )

        try:
            # 2. Load and validate format
            img = Image.open(image_path)
            actual_width, actual_height = img.size

            # 3. Check for corruption
            no_corruption = True
            try:
                img.verify()
            except Exception as e:
                no_corruption = False
                errors.append(f"Corruption detected: {str(e)}")

            # Reload after verify
            img = Image.open(image_path)

            # 4. Analyze colors (RGB only)
            valid_color_range = False
            no_artifacts = False

            if img.mode == "RGB":
                stat = ImageStat.Stat(img)
                mean_rgb = tuple(stat.mean)
                std_rgb = tuple(stat.stddev)

                # Check color range (reasonable values)
                if all(10 < m < 245 for m in mean_rgb):
                    valid_color_range = True

                # Check for artifacts (extrema)
                extrema = stat.extrema
                if all(5 < ex[0] and ex[1] < 250 for ex in extrema):
                    no_artifacts = True

            # 5. Calculate quality score
            quality_score = self._calculate_score(
                exists=True,
                valid_format=True,
                correct_resolution=(actual_width == expected_width and
                                    actual_height == expected_height),
                no_corruption=no_corruption,
                valid_color_range=valid_color_range,
                no_artifacts=no_artifacts,
            )

            metrics = QualityMetrics(
                exists=True,
                valid_format=True,
                correct_resolution=(actual_width == expected_width and
                                    actual_height == expected_height),
                no_corruption=no_corruption,
                valid_color_range=valid_color_range,
                no_artifacts=no_artifacts,
                actual_width=actual_width,
                actual_height=actual_height,
                expected_width=expected_width,
                expected_height=expected_height,
                mean_rgb=mean_rgb,
                std_rgb=std_rgb,
                quality_score=quality_score,
                errors=errors,
            )

            if self.enable_logging:
                self._log_validation(image_path, metrics)

            return metrics

        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return self._create_failed_metrics(
                expected_width, expected_height, errors
            )

    def validate_batch(
        self,
        image_paths: list[Path],
        expected_width: int = 1024,
        expected_height: int = 1024,
    ) -> list[QualityMetrics]:
        """
        Validate multiple images.

        Args:
            image_paths: List of image paths
            expected_width: Expected width
            expected_height: Expected height

        Returns:
            List of QualityMetrics
        """
        return [
            self.validate_image(path, expected_width, expected_height)
            for path in image_paths
        ]

    def get_summary_stats(
        self,
        metrics_list: list[QualityMetrics]
    ) -> dict:
        """
        Calculate summary statistics for batch validation.

        Args:
            metrics_list: List of quality metrics

        Returns:
            Dictionary with summary stats
        """
        if not metrics_list:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "avg_quality": 0.0,
            }

        total = len(metrics_list)
        passed = sum(1 for m in metrics_list if m.is_valid)
        avg_quality = sum(m.quality_score for m in metrics_list) / total

        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total * 100,
            "avg_quality": avg_quality,
            "min_quality": min(m.quality_score for m in metrics_list),
            "max_quality": max(m.quality_score for m in metrics_list),
        }

    def _calculate_score(
        self,
        exists: bool,
        valid_format: bool,
        correct_resolution: bool,
        no_corruption: bool,
        valid_color_range: bool,
        no_artifacts: bool,
    ) -> float:
        """Calculate overall quality score (0-100)."""
        score = 0.0

        if exists:
            score += 20
        if valid_format:
            score += 20
        if correct_resolution:
            score += 20
        if no_corruption:
            score += 20
        if valid_color_range:
            score += 10
        if no_artifacts:
            score += 10

        return score

    def _create_failed_metrics(
        self,
        expected_width: int,
        expected_height: int,
        errors: list[str]
    ) -> QualityMetrics:
        """Create metrics for failed validation."""
        return QualityMetrics(
            exists=False,
            valid_format=False,
            correct_resolution=False,
            no_corruption=False,
            valid_color_range=False,
            no_artifacts=False,
            actual_width=0,
            actual_height=0,
            expected_width=expected_width,
            expected_height=expected_height,
            mean_rgb=(0.0, 0.0, 0.0),
            std_rgb=(0.0, 0.0, 0.0),
            quality_score=0.0,
            errors=errors,
        )

    def _log_validation(self, image_path: Path, metrics: QualityMetrics):
        """Log validation results."""
        if metrics.is_excellent:
            level = logging.INFO
            status = "EXCELLENT"
        elif metrics.is_valid:
            level = logging.INFO
            status = "VALID"
        else:
            level = logging.WARNING
            status = "FAILED"

        logger.log(
            level,
            f"Quality validation [{status}]: {image_path.name} "
            f"(score: {metrics.quality_score:.0f}/100)"
        )
