"""Resolution value object for image dimensions.

This module provides a type-safe Resolution class to replace tuple[int, int] usage
throughout the codebase.
"""

from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True)
class Resolution:
    """Represents image resolution with validation.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.

    Example:
        >>> res = Resolution(1024, 768)
        >>> print(res.width, res.height)
        1024 768
        >>> print(res.aspect_ratio)
        1.333...
        >>> print(res.total_pixels)
        786432
    """

    width: int
    height: int

    def __post_init__(self) -> None:
        """Validate resolution dimensions."""
        if self.width <= 0:
            raise ValueError(f"Width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"Height must be positive, got {self.height}")
        if self.width % 8 != 0:
            raise ValueError(f"Width must be divisible by 8, got {self.width}")
        if self.height % 8 != 0:
            raise ValueError(f"Height must be divisible by 8, got {self.height}")

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width/height)."""
        return self.width / self.height

    @property
    def total_pixels(self) -> int:
        """Calculate total number of pixels."""
        return self.width * self.height

    @property
    def megapixels(self) -> float:
        """Calculate megapixels."""
        return self.total_pixels / 1_000_000

    def is_landscape(self) -> bool:
        """Check if resolution is landscape orientation."""
        return self.width > self.height

    def is_portrait(self) -> bool:
        """Check if resolution is portrait orientation."""
        return self.height > self.width

    def is_square(self) -> bool:
        """Check if resolution is square."""
        return self.width == self.height

    def scale(self, factor: float) -> Self:
        """Scale resolution by a factor.

        Args:
            factor: Scaling factor (e.g., 2.0 doubles resolution).

        Returns:
            New Resolution instance with scaled dimensions.
        """
        new_width = int(self.width * factor)
        new_height = int(self.height * factor)
        # Round to nearest multiple of 8
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        return Resolution(new_width, new_height)

    def fit_within(self, max_resolution: Self) -> Self:
        """Scale down to fit within max resolution while maintaining aspect ratio.

        Args:
            max_resolution: Maximum allowed resolution.

        Returns:
            New Resolution that fits within constraints.
        """
        if self.width <= max_resolution.width and self.height <= max_resolution.height:
            return self

        width_scale = max_resolution.width / self.width
        height_scale = max_resolution.height / self.height
        scale_factor = min(width_scale, height_scale)

        return self.scale(scale_factor)


__all__ = ["Resolution"]
