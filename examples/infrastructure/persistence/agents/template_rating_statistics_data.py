"""Template rating statistics data transfer object."""

from dataclasses import dataclass


@dataclass
class TemplateRatingStatisticsData:
    """Data transfer object for template rating statistics."""

    avg_rating: float
    total_count: int
    verified_count: int
    stars_1: int = 0
    stars_2: int = 0
    stars_3: int = 0
    stars_4: int = 0
    stars_5: int = 0
