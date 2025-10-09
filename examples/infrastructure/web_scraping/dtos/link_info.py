"""Link information data class."""

from dataclasses import dataclass


@dataclass
class LinkInfo:
    """Information about extracted links."""

    url: str
    text: str
    link_type: str  # internal, external, mailto, etc
    depth_level: int = 0
    is_scraped: bool = False
