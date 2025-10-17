"""
Image metadata embedding and naming system.

Provides comprehensive metadata embedding in generated images:
- PNG text chunks (full configuration)
- EXIF metadata (standard fields)
- Standardized naming with timestamps and GUIDs
- Secure and anonymous (no user tracking)

Features:
- Full generation config embedded as PNG tEXt chunk
- EXIF tags for standard metadata
- GUID-based unique identifiers
- Timestamp-based naming with ISO 8601
- Optional prompt excerpt in filename
- Metadata extraction from saved images
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PIL import Image
from PIL.ExifTags import TAGS
from PIL.PngImagePlugin import PngInfo

from ml_lib.diffusion.domain.services.image_naming import ImageNamingConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoRAMetadataEntry:
    """LoRA metadata entry for image metadata."""

    name: str
    weight: float
    source: str = "unknown"


@dataclass
class ImageMetadataEmbedding:
    """
    Complete metadata for embedding in generated images.

    Includes all information needed to reproduce the generation.
    Designed to be secure and anonymous - no user tracking.
    """

    # Generation ID (unique)
    generation_id: str
    """Unique GUID for this generation."""

    generation_timestamp: str
    """ISO 8601 timestamp when generated (UTC)."""

    # Prompt
    prompt: str
    """Full positive prompt."""

    negative_prompt: str
    """Full negative prompt."""

    # Core parameters
    seed: int
    """Random seed used."""

    steps: int
    """Number of diffusion steps."""

    cfg_scale: float
    """Classifier-free guidance scale."""

    width: int
    """Image width in pixels."""

    height: int
    """Image height in pixels."""

    sampler: str
    """Sampler name."""

    scheduler: str = "default"
    """Scheduler name."""

    clip_skip: int = 2
    """CLIP skip value."""

    # Model information
    base_model_id: str = "unknown"
    """Base model identifier or path."""

    base_model_architecture: str = "unknown"
    """Architecture: SD1.5, SDXL, Flux, etc."""

    vae_model: Optional[str] = None
    """VAE model if custom."""

    # LoRAs
    loras_used: list[LoRAMetadataEntry] = field(default_factory=list)
    """List of LoRAs used in generation."""

    # Performance
    generation_time_seconds: Optional[float] = None
    """Time taken to generate."""

    peak_vram_gb: Optional[float] = None
    """Peak VRAM usage."""

    # Pipeline
    pipeline_type: str = "intelligent"
    """Pipeline type: intelligent, standard, etc."""

    pipeline_version: str = "1.0.0"
    """Version of pipeline used."""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert LoRAMetadataEntry objects to dicts for JSON serialization
        if "loras_used" in data:
            data["loras_used"] = [
                {"name": lora.name, "weight": lora.weight, "source": lora.source}
                if isinstance(lora, LoRAMetadataEntry)
                else lora
                for lora in self.loras_used
            ]
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "ImageMetadataEmbedding":
        """Create from dictionary."""
        # Convert loras_used dicts to LoRAMetadataEntry objects
        if "loras_used" in data and data["loras_used"]:
            data["loras_used"] = [
                LoRAMetadataEntry(**lora) if isinstance(lora, dict) else lora
                for lora in data["loras_used"]
            ]
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "ImageMetadataEmbedding":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


class ImageMetadataWriter:
    """
    Writes metadata to images securely and anonymously.

    Supports:
    - PNG tEXt chunks (full config as JSON)
    - EXIF metadata (standard fields)
    - Standardized naming
    """

    def __init__(self, naming_config: Optional[ImageNamingConfig] = None):
        """
        Initialize metadata writer.

        Args:
            naming_config: Naming configuration (default: standard)
        """
        self.naming_config = naming_config or ImageNamingConfig.standard()

    def generate_filename(
        self,
        metadata: ImageMetadataEmbedding,
        custom_prefix: Optional[str] = None,
    ) -> str:
        """
        Generate standardized filename from metadata.

        Args:
            metadata: Generation metadata
            custom_prefix: Optional custom prefix

        Returns:
            Filename (without path)

        Examples:
            Standard: "20250111_143022_a3f2e9d4.png"
            Descriptive: "20250111_143022_beautiful-sunset_a3f2e9d4.png"
            GUID-only: "a3f2e9d4-b2c1-4a8e-9f3d-1e2a4b5c6d7e.png"
        """
        parts = []

        # Prefix
        if custom_prefix:
            parts.append(self._sanitize_for_filename(custom_prefix))

        # Timestamp
        if self.naming_config.include_timestamp:
            # Parse ISO 8601 timestamp and format
            try:
                dt = datetime.fromisoformat(metadata.generation_timestamp)
                timestamp = dt.strftime(self.naming_config.timestamp_format)
                parts.append(timestamp)
            except Exception:
                # Fallback to current time
                timestamp = datetime.now(timezone.utc).strftime(
                    self.naming_config.timestamp_format
                )
                parts.append(timestamp)

        # Prompt excerpt
        if self.naming_config.include_prompt_excerpt:
            excerpt = self._sanitize_prompt_excerpt(
                metadata.prompt, self.naming_config.prompt_excerpt_length
            )
            if excerpt:
                parts.append(excerpt)

        # GUID
        if self.naming_config.include_guid:
            # Use first 8 chars of generation ID for brevity
            guid_short = metadata.generation_id.split("-")[0]
            parts.append(guid_short)

        # Join and add extension
        filename = self.naming_config.separator.join(parts)
        return f"{filename}.{self.naming_config.extension}"

    def _sanitize_for_filename(self, text: str) -> str:
        """Sanitize text for use in filename."""
        # Replace invalid chars with underscores
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            text = text.replace(char, "_")
        # Remove multiple consecutive underscores
        while "__" in text:
            text = text.replace("__", "_")
        return text.strip("_")

    def _sanitize_prompt_excerpt(self, prompt: str, max_length: int) -> str:
        """
        Create sanitized excerpt from prompt.

        Takes first N characters, removes special chars, converts to lowercase.
        """
        # Take first max_length chars
        excerpt = prompt[:max_length]

        # Remove special characters, keep only alphanumeric and spaces
        excerpt = "".join(c if c.isalnum() or c.isspace() else "" for c in excerpt)

        # Convert to lowercase and replace spaces with hyphens
        excerpt = excerpt.lower().strip()
        excerpt = "-".join(excerpt.split())

        return excerpt

    def save_with_metadata(
        self,
        image: Image.Image,
        metadata: ImageMetadataEmbedding,
        output_dir: Path | str,
        filename: Optional[str] = None,
        embed_full_json: bool = True,
        embed_exif: bool = True,
        save_sidecar_json: bool = False,
    ) -> Path:
        """
        Save image with embedded metadata.

        Args:
            image: PIL Image to save
            metadata: Metadata to embed
            output_dir: Output directory
            filename: Custom filename (None = auto-generate)
            embed_full_json: Embed full JSON as PNG tEXt chunk
            embed_exif: Embed standard EXIF fields
            save_sidecar_json: Also save metadata as separate .json file

        Returns:
            Path to saved image

        Example:
            >>> writer = ImageMetadataWriter()
            >>> path = writer.save_with_metadata(
            ...     image=my_image,
            ...     metadata=my_metadata,
            ...     output_dir="/outputs",
            ... )
            >>> print(path)
            /outputs/20250111_143022_a3f2e9d4.png
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename if not provided
        if filename is None:
            filename = self.generate_filename(metadata)

        output_path = output_dir / filename

        # Create PNG info
        pnginfo = PngInfo()

        # Embed full JSON as tEXt chunk
        if embed_full_json:
            # Main metadata chunk
            pnginfo.add_text("ml_lib_metadata", metadata.to_json())

            # Also add individual fields for easier parsing
            pnginfo.add_text("generation_id", metadata.generation_id)
            pnginfo.add_text("prompt", metadata.prompt)
            pnginfo.add_text("negative_prompt", metadata.negative_prompt)
            pnginfo.add_text("seed", str(metadata.seed))
            pnginfo.add_text("steps", str(metadata.steps))
            pnginfo.add_text("cfg_scale", str(metadata.cfg_scale))
            pnginfo.add_text("sampler", metadata.sampler)
            pnginfo.add_text("base_model", metadata.base_model_id)

        # Embed EXIF metadata
        if embed_exif:
            exif_data = self._create_exif_data(metadata)
            image.save(output_path, pnginfo=pnginfo, exif=exif_data)
        else:
            image.save(output_path, pnginfo=pnginfo)

        # Save sidecar JSON
        if save_sidecar_json:
            sidecar_path = output_path.with_suffix(".metadata.json")
            with open(sidecar_path, "w") as f:
                f.write(metadata.to_json())

        return output_path

    def _create_exif_data(self, metadata: ImageMetadataEmbedding) -> bytes:
        """
        Create EXIF data from metadata.

        Uses standard EXIF tags where applicable:
        - ImageDescription: Prompt
        - UserComment: Full generation config
        - Software: Pipeline info
        - DateTime: Generation timestamp
        """
        exif = Image.Exif()

        # Standard tags (using tag IDs directly for compatibility)
        # 0x010e = ImageDescription
        exif[0x010E] = f"Prompt: {metadata.prompt[:200]}"

        # 0x0131 = Software
        exif[0x0131] = f"{metadata.pipeline_type} v{metadata.pipeline_version}"

        # 0x0132 = DateTime
        try:
            dt = datetime.fromisoformat(metadata.generation_timestamp)
            exif[0x0132] = dt.strftime("%Y:%m:%d %H:%M:%S")
        except Exception as e:
            # Log parsing error but continue without timestamp in EXIF
            logger.debug(f"Could not parse timestamp for EXIF: {e}")

        # 0x9286 = UserComment (full metadata as JSON)
        exif[0x9286] = metadata.to_json()

        return exif.tobytes()

    def extract_metadata(self, image_path: Path | str) -> Optional[ImageMetadataEmbedding]:
        """
        Extract metadata from saved image.

        Args:
            image_path: Path to image file

        Returns:
            Extracted metadata or None if not found

        Example:
            >>> writer = ImageMetadataWriter()
            >>> metadata = writer.extract_metadata("/outputs/image.png")
            >>> print(metadata.prompt)
            a beautiful sunset
        """
        image_path = Path(image_path)

        if not image_path.exists():
            return None

        try:
            with Image.open(image_path) as img:
                # Try to extract from PNG tEXt chunks first
                if hasattr(img, "text") and "ml_lib_metadata" in img.text:
                    json_str = img.text["ml_lib_metadata"]
                    return ImageMetadataEmbedding.from_json(json_str)

                # Try to extract from EXIF UserComment
                if hasattr(img, "getexif"):
                    exif = img.getexif()
                    if 0x9286 in exif:  # UserComment
                        json_str = exif[0x9286]
                        return ImageMetadataEmbedding.from_json(json_str)

        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return None

        return None


def create_generation_id() -> str:
    """
    Create a unique generation ID.

    Returns:
        UUID v4 as string

    Example:
        >>> gen_id = create_generation_id()
        >>> print(gen_id)
        a3f2e9d4-b2c1-4a8e-9f3d-1e2a4b5c6d7e
    """
    return str(uuid.uuid4())


def create_timestamp() -> str:
    """
    Create ISO 8601 timestamp (UTC).

    Returns:
        ISO 8601 timestamp string

    Example:
        >>> timestamp = create_timestamp()
        >>> print(timestamp)
        2025-01-11T14:30:22.123456+00:00
    """
    return datetime.now(timezone.utc).isoformat()
