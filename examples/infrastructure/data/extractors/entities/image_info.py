from dataclasses import dataclass


@dataclass
class ImageInfo:
    """Información de una imagen extraída."""

    index: int
    image_type: str
    width: int | None = None
    height: int | None = None
    format_type: str | None = None
    alt_text: str | None = None
    url: str | None = None
    position: int = 0
    data_available: bool = False
