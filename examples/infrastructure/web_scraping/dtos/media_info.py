"""Media information data class."""

from dataclasses import dataclass


@dataclass
class MediaInfo:
    """Information about extracted media."""

    url: str
    alt_text: str
    media_type: str  # image, video, audio
    file_size_estimate: int | None = None
