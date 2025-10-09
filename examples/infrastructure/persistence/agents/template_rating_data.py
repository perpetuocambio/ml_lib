"""Data structure for template rating persistence."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TemplateRatingData:
    """Data structure for template rating database records."""

    rating_id: str
    template_id: str
    user_id: str
    rating: int  # 1-5 stars
    review_text: str | None
    created_at: str
    is_verified_user: bool
