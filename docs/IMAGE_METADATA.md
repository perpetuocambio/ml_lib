# Image Metadata & Naming System

Complete guide to the metadata embedding and naming system for generated images.

## Overview

The metadata system provides:

- **Full configuration embedding** - All generation parameters stored in images
- **Standardized naming** - Timestamp + GUID based filenames
- **Privacy-focused** - No user tracking, fully anonymous
- **Multiple formats** - PNG tEXt chunks + EXIF metadata
- **Easy extraction** - Read metadata from any saved image
- **Reproducibility** - Complete information to reproduce any generation

## Quick Start

### Basic Usage

```python
from ml_lib.diffusion.intelligent.pipeline.services.image_metadata import (
    ImageMetadataWriter,
    ImageMetadataEmbedding,
    create_generation_id,
    create_timestamp,
)

# Create metadata
metadata = ImageMetadataEmbedding(
    generation_id=create_generation_id(),
    generation_timestamp=create_timestamp(),
    prompt="a beautiful sunset over mountains",
    negative_prompt="low quality, blurry",
    seed=42,
    steps=30,
    cfg_scale=7.5,
    width=1024,
    height=1024,
    sampler="DPM++ 2M",
    base_model_id="sdxl-base-1.0",
    base_model_architecture="SDXL",
)

# Save image with metadata
writer = ImageMetadataWriter()
output_path = writer.save_with_metadata(
    image=my_image,
    metadata=metadata,
    output_dir="/outputs",
)

# Result: /outputs/20250111_143022_a3f2e9d4.png
```

### Using with GenerationResult

The metadata system is integrated into `GenerationResult`:

```python
# Method 1: Auto-generated naming
result.save("/outputs", use_auto_naming=True)
# → /outputs/20250111_143022_a3f2e9d4.png

# Method 2: Descriptive naming (includes prompt excerpt)
from ml_lib.diffusion.intelligent.pipeline.services import ImageNamingConfig

result.save(
    "/outputs",
    use_auto_naming=True,
    naming_config=ImageNamingConfig.descriptive()
)
# → /outputs/20250111_143022_beautiful-sunset_a3f2e9d4.png

# Method 3: Custom filename
result.save("/outputs/my_custom_name.png")
# → /outputs/my_custom_name.png
```

All methods automatically embed full metadata!

## Naming Conventions

### Standard Naming (Default)

Format: `YYYYMMDD_HHMMSS_GUID.png`

Example: `20250111_143022_a3f2e9d4.png`

```python
config = ImageNamingConfig.standard()
```

**Components:**
- Timestamp: `20250111_143022` (UTC)
- GUID: `a3f2e9d4` (first 8 chars of UUID)

**Benefits:**
- Chronological sorting
- Guaranteed uniqueness
- Human-readable timestamp

### Descriptive Naming

Format: `YYYYMMDD_HHMMSS_prompt-excerpt_GUID.png`

Example: `20250111_143022_beautiful-sunset_a3f2e9d4.png`

```python
config = ImageNamingConfig.descriptive()
```

**Components:**
- Timestamp: `20250111_143022`
- Prompt excerpt: `beautiful-sunset` (sanitized, first 30 chars)
- GUID: `a3f2e9d4`

**Benefits:**
- Human-readable context
- Easy to identify images
- Still guaranteed unique

### GUID-Only Naming (Anonymous)

Format: `GUID.png`

Example: `a3f2e9d4-b2c1-4a8e-9f3d-1e2a4b5c6d7e.png`

```python
config = ImageNamingConfig.guid_only()
```

**Benefits:**
- Maximum privacy (no timestamp in filename)
- No prompt excerpt in filename
- Fully anonymous
- Metadata still embedded inside image

### Custom Naming

```python
config = ImageNamingConfig(
    include_timestamp=True,
    include_guid=True,
    include_prompt_excerpt=True,
    prompt_excerpt_length=20,        # Custom length
    timestamp_format="%Y%m%d",       # Date only
    separator="-",                    # Use hyphens
    extension="png",
)

# Result: 20250111-fantasy-landscape-a3f2e9d4.png
```

## Metadata Embedding

### What's Embedded

Full generation configuration is embedded in multiple formats:

1. **PNG tEXt Chunks** (Primary)
   - `ml_lib_metadata`: Complete JSON with all parameters
   - Individual text chunks: `prompt`, `seed`, `steps`, etc.

2. **EXIF Metadata** (Standard)
   - ImageDescription: Prompt
   - Software: Pipeline version
   - DateTime: Generation timestamp
   - UserComment: Full JSON configuration

### Metadata Fields

```python
@dataclass
class ImageMetadataEmbedding:
    # Identification
    generation_id: str              # Unique GUID
    generation_timestamp: str       # ISO 8601 timestamp (UTC)

    # Prompt
    prompt: str                     # Full positive prompt
    negative_prompt: str            # Full negative prompt

    # Core parameters
    seed: int                       # Random seed
    steps: int                      # Diffusion steps
    cfg_scale: float                # CFG scale
    width: int                      # Image width
    height: int                     # Image height
    sampler: str                    # Sampler name
    scheduler: str                  # Scheduler name
    clip_skip: int                  # CLIP skip value

    # Model information
    base_model_id: str              # Model ID or path
    base_model_architecture: str    # SD1.5, SDXL, Flux, etc.
    vae_model: Optional[str]        # Custom VAE if used

    # LoRAs
    loras_used: list[dict]          # [{"name": "...", "weight": 0.8}]

    # Performance
    generation_time_seconds: float  # Time taken
    peak_vram_gb: float             # Peak VRAM usage

    # Pipeline
    pipeline_type: str              # Pipeline type
    pipeline_version: str           # Pipeline version
```

### Privacy & Security

**What's Included:**
- Generation parameters (for reproducibility)
- Model information (for tracking)
- Performance metrics (for optimization)
- Timestamp and GUID (for uniqueness)

**What's NOT Included:**
- User identity or personal information
- API keys or credentials
- System paths or environment details
- Network or location data

**Security Features:**
- No code execution in metadata
- Standard PNG/EXIF formats only
- Sanitized filenames (no special characters)
- Read-only after embedding

## Extracting Metadata

### From Saved Images

```python
writer = ImageMetadataWriter()
metadata = writer.extract_metadata("/outputs/image.png")

if metadata:
    print(f"Prompt: {metadata.prompt}")
    print(f"Seed: {metadata.seed}")
    print(f"Model: {metadata.base_model_architecture}")
    print(f"LoRAs: {len(metadata.loras_used)}")

    # Full reproduction possible!
    # All parameters needed to recreate the image
```

### What Can Be Extracted

- Complete generation configuration
- All model information (base model, LoRAs, VAE)
- Exact parameters used (seed, steps, CFG, etc.)
- Performance metrics
- Generation timestamp

**Use Cases:**
- Reproduce successful generations
- Debug issues
- Track model performance
- Build generation history
- Share configurations

## Advanced Usage

### Sidecar JSON Files

Save metadata as a separate `.metadata.json` file alongside the image:

```python
output_path = writer.save_with_metadata(
    image=my_image,
    metadata=metadata,
    output_dir="/outputs",
    save_sidecar_json=True,  # Also save as JSON
)

# Creates:
# - /outputs/20250111_143022_a3f2e9d4.png
# - /outputs/20250111_143022_a3f2e9d4.metadata.json
```

**Benefits:**
- Easy to read/edit with text editor
- Can be indexed by search tools
- Survives image format conversions
- Metadata in image AND separate file

### Custom Filename Generation

```python
writer = ImageMetadataWriter()

# Generate filename only (don't save yet)
filename = writer.generate_filename(
    metadata=metadata,
    custom_prefix="test",  # Optional prefix
)

# Result: "test_20250111_143022_a3f2e9d4.png"
```

### Converting Legacy Metadata

Convert old `GenerationMetadata` to new `ImageMetadataEmbedding`:

```python
# Old format
old_metadata = GenerationMetadata(
    prompt="...",
    seed=42,
    # ... other fields
)

# Convert to new format
new_metadata = old_metadata.to_image_metadata(
    base_model_architecture="SDXL",
    scheduler="karras",
)

# Now has all enhanced features
```

## API Reference

### ImageMetadataWriter

Main class for writing and reading metadata.

```python
class ImageMetadataWriter:
    def __init__(self, naming_config: Optional[ImageNamingConfig] = None)

    def save_with_metadata(
        self,
        image: Image.Image,
        metadata: ImageMetadataEmbedding,
        output_dir: Path | str,
        filename: Optional[str] = None,
        embed_full_json: bool = True,
        embed_exif: bool = True,
        save_sidecar_json: bool = False,
    ) -> Path

    def generate_filename(
        self,
        metadata: ImageMetadataEmbedding,
        custom_prefix: Optional[str] = None,
    ) -> str

    def extract_metadata(
        self,
        image_path: Path | str
    ) -> Optional[ImageMetadataEmbedding]
```

### ImageNamingConfig

Configuration for filename generation.

```python
@dataclass
class ImageNamingConfig:
    include_timestamp: bool = True
    include_guid: bool = True
    include_prompt_excerpt: bool = False
    prompt_excerpt_length: int = 30
    timestamp_format: str = "%Y%m%d_%H%M%S"
    separator: str = "_"
    extension: str = "png"

    @classmethod
    def standard(cls) -> "ImageNamingConfig"

    @classmethod
    def descriptive(cls) -> "ImageNamingConfig"

    @classmethod
    def guid_only(cls) -> "ImageNamingConfig"
```

### Utility Functions

```python
def create_generation_id() -> str:
    """Create unique UUID v4."""

def create_timestamp() -> str:
    """Create ISO 8601 timestamp (UTC)."""
```

## Examples

See `examples/metadata_examples.py` for comprehensive examples including:

1. Basic metadata embedding
2. Descriptive naming with prompt excerpt
3. GUID-only naming for anonymity
4. Custom naming configurations
5. Metadata extraction from images
6. Sidecar JSON files
7. Privacy and security features
8. Integration with GenerationResult

## Best Practices

### Naming

1. **Use standard naming by default** - Good balance of readability and privacy
2. **Use descriptive naming for curated collections** - Easier to browse
3. **Use GUID-only for sensitive content** - Maximum privacy
4. **Avoid special characters in prompts** - They're sanitized anyway

### Metadata

1. **Always embed metadata** - Future you will thank you
2. **Include model architecture** - Critical for reproduction
3. **Track LoRA weights** - Important for fine-tuning results
4. **Save performance metrics** - Useful for optimization

### Privacy

1. **Never include personal info in prompts** - It gets embedded
2. **Use GUID-only naming for privacy** - No timestamp leaks
3. **Review metadata before sharing** - Know what's embedded
4. **Consider sidecar JSON for archiving** - Easier to index/search

### Organization

1. **Use timestamp-based naming** - Easy chronological sorting
2. **Organize by date subdirectories** - `YYYY/MM/DD/images/`
3. **Keep metadata embedded** - Never gets separated
4. **Consider sidecar JSON for databases** - Easier to query

## Migration Guide

### From Old System

The old `GenerationResult.save()` still works but uses legacy metadata format:

```python
# Old way (still works)
result.save("/outputs/image.png", save_metadata=True)
# → Metadata as single JSON chunk

# New way (enhanced)
result.save("/outputs", use_auto_naming=True)
# → Full metadata with EXIF + auto-naming
```

### Updating Existing Code

```python
# Before
result.save(output_path, save_metadata=True)

# After (auto-naming)
result.save(output_dir, use_auto_naming=True)

# After (custom naming with metadata)
result.save(
    output_path,
    save_metadata=True,
    naming_config=ImageNamingConfig.standard()
)
```

## Troubleshooting

### Metadata Not Extracting

- Ensure image is PNG format (JPEG loses tEXt chunks)
- Check if image was saved with `save_metadata=True`
- Try extracting from EXIF if PNG chunks missing

### Filenames Look Strange

- Prompt excerpts are sanitized (alphanumeric only)
- Special characters become hyphens or removed
- This is intentional for filesystem compatibility

### Large Metadata

- Metadata is typically <5KB per image
- Negligible compared to image size (MB)
- PNG supports unlimited tEXt chunks

## Future Enhancements

Potential future additions:

- [ ] Support for video metadata (MP4, etc.)
- [ ] Metadata database for faster searching
- [ ] Automatic tagging based on prompt
- [ ] Integration with image galleries
- [ ] Metadata templates for common workflows
- [ ] Batch metadata extraction tools

## Related Documentation

- [Intelligent Generation Guide](INTELLIGENT_GENERATION.md)
- [Pipeline Architecture](PIPELINE_ARCHITECTURE.md)
- [Model Orchestration](MODEL_ORCHESTRATION.md)
