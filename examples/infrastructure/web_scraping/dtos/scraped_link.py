from dataclasses import dataclass


@dataclass(frozen=True)
class ScrapedLink:
    """Represents a scraped hyperlink with its URL and text."""

    url: str
    text: str
