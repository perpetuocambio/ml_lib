from enum import Enum


class SortBy(Enum):
    """Sort options for model search."""

    RELEVANCE = "relevance"
    POPULARITY = "popularity"
    DOWNLOADS = "downloads"
    RATING = "rating"
    NEWEST = "newest"
    UPDATED = "updated"
