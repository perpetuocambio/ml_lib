"""
Examples of using the metadata and naming system.

This demonstrates:
1. Automatic naming with timestamps and GUIDs
2. Embedding metadata in generated images
3. Extracting metadata from saved images
4. Different naming configurations
5. Privacy and security considerations
"""

from pathlib import Path
from PIL import Image

from ml_lib.diffusion.intelligent.pipeline.services.image_metadata import (
    ImageMetadataWriter,
    ImageMetadataEmbedding,
    ImageNamingConfig,
    create_generation_id,
    create_timestamp,
)


def example_1_basic_metadata():
    """Example 1: Basic metadata embedding with standard naming."""
    print("\n=== Example 1: Basic Metadata Embedding ===\n")

    # Create sample metadata
    metadata = ImageMetadataEmbedding(
        generation_id=create_generation_id(),
        generation_timestamp=create_timestamp(),
        prompt="a beautiful sunset over mountains",
        negative_prompt="worst quality, low quality, blurry",
        seed=42,
        steps=30,
        cfg_scale=7.5,
        width=1024,
        height=1024,
        sampler="DPM++ 2M",
        scheduler="karras",
        base_model_id="stabilityai/stable-diffusion-xl-base-1.0",
        base_model_architecture="SDXL",
        loras_used=[
            {"name": "landscape-lora", "weight": 0.8, "source": "civitai"},
        ],
        generation_time_seconds=15.3,
        peak_vram_gb=8.2,
    )

    # Create dummy image for testing
    dummy_image = Image.new("RGB", (1024, 1024), color=(255, 128, 0))

    # Initialize metadata writer with standard naming
    writer = ImageMetadataWriter(naming_config=ImageNamingConfig.standard())

    # Save with metadata
    output_path = writer.save_with_metadata(
        image=dummy_image,
        metadata=metadata,
        output_dir=Path("./outputs"),
        embed_full_json=True,
        embed_exif=True,
    )

    print(f"✓ Image saved to: {output_path}")
    print(f"  Filename format: YYYYMMDD_HHMMSS_GUID.png")
    print(f"  Example: {output_path.name}")
    print(f"\n✓ Metadata embedded:")
    print(f"  - PNG tEXt chunks: Full JSON configuration")
    print(f"  - EXIF tags: Standard metadata fields")
    print(f"  - Prompt: {metadata.prompt[:50]}...")
    print(f"  - Seed: {metadata.seed}")
    print(f"  - Model: {metadata.base_model_architecture}")


def example_2_descriptive_naming():
    """Example 2: Descriptive naming with prompt excerpt."""
    print("\n=== Example 2: Descriptive Naming ===\n")

    metadata = ImageMetadataEmbedding(
        generation_id=create_generation_id(),
        generation_timestamp=create_timestamp(),
        prompt="cyberpunk city at night with neon lights and rain",
        negative_prompt="low quality",
        seed=123,
        steps=40,
        cfg_scale=8.0,
        width=1024,
        height=1024,
        sampler="Euler a",
        base_model_id="flux-dev",
        base_model_architecture="Flux",
    )

    dummy_image = Image.new("RGB", (1024, 1024), color=(0, 128, 255))

    # Use descriptive naming (includes prompt excerpt)
    writer = ImageMetadataWriter(naming_config=ImageNamingConfig.descriptive())

    output_path = writer.save_with_metadata(
        image=dummy_image,
        metadata=metadata,
        output_dir=Path("./outputs"),
    )

    print(f"✓ Image saved to: {output_path}")
    print(f"  Filename format: YYYYMMDD_HHMMSS_prompt-excerpt_GUID.png")
    print(f"  Example: {output_path.name}")
    print(f"\n✓ Descriptive naming includes:")
    print(f"  - Timestamp: Easy chronological sorting")
    print(f"  - Prompt excerpt: Human-readable context")
    print(f"  - GUID: Guaranteed uniqueness")


def example_3_guid_only():
    """Example 3: GUID-only naming for maximum anonymity."""
    print("\n=== Example 3: GUID-Only Naming (Anonymous) ===\n")

    metadata = ImageMetadataEmbedding(
        generation_id=create_generation_id(),
        generation_timestamp=create_timestamp(),
        prompt="portrait of a person",
        negative_prompt="low quality",
        seed=456,
        steps=25,
        cfg_scale=7.0,
        width=512,
        height=768,
        sampler="DPM++ SDE",
        base_model_id="sd-1.5",
        base_model_architecture="SD1.5",
    )

    dummy_image = Image.new("RGB", (512, 768), color=(128, 0, 255))

    # Use GUID-only naming (most anonymous)
    writer = ImageMetadataWriter(naming_config=ImageNamingConfig.guid_only())

    output_path = writer.save_with_metadata(
        image=dummy_image,
        metadata=metadata,
        output_dir=Path("./outputs"),
    )

    print(f"✓ Image saved to: {output_path}")
    print(f"  Filename format: GUID.png")
    print(f"  Example: {output_path.name}")
    print(f"\n✓ Maximum privacy:")
    print(f"  - No timestamp in filename")
    print(f"  - No prompt excerpt in filename")
    print(f"  - Fully anonymous filename")
    print(f"  - Metadata still embedded inside image")


def example_4_custom_naming():
    """Example 4: Custom naming configuration."""
    print("\n=== Example 4: Custom Naming Configuration ===\n")

    metadata = ImageMetadataEmbedding(
        generation_id=create_generation_id(),
        generation_timestamp=create_timestamp(),
        prompt="fantasy landscape with castle",
        negative_prompt="low quality",
        seed=789,
        steps=35,
        cfg_scale=7.5,
        width=1024,
        height=1024,
        sampler="DPM++ 2M Karras",
        base_model_id="dreamshaper-8",
        base_model_architecture="SD1.5",
    )

    dummy_image = Image.new("RGB", (1024, 1024), color=(255, 255, 0))

    # Create custom naming config
    custom_config = ImageNamingConfig(
        include_timestamp=True,
        include_guid=True,
        include_prompt_excerpt=True,
        prompt_excerpt_length=20,  # Shorter excerpt
        timestamp_format="%Y%m%d",  # Date only, no time
        separator="-",  # Use hyphens instead of underscores
        extension="png",
    )

    writer = ImageMetadataWriter(naming_config=custom_config)

    output_path = writer.save_with_metadata(
        image=dummy_image,
        metadata=metadata,
        output_dir=Path("./outputs"),
    )

    print(f"✓ Image saved to: {output_path}")
    print(f"  Filename format: YYYYMMDD-prompt-excerpt-GUID.png")
    print(f"  Example: {output_path.name}")
    print(f"\n✓ Custom configuration:")
    print(f"  - Date format: YYYYMMDD (no time)")
    print(f"  - Separator: hyphens")
    print(f"  - Prompt excerpt: 20 chars")


def example_5_metadata_extraction():
    """Example 5: Extracting metadata from saved images."""
    print("\n=== Example 5: Metadata Extraction ===\n")

    # First, create and save an image with metadata
    metadata = ImageMetadataEmbedding(
        generation_id=create_generation_id(),
        generation_timestamp=create_timestamp(),
        prompt="anime girl with pink hair",
        negative_prompt="worst quality, low quality",
        seed=999,
        steps=28,
        cfg_scale=7.0,
        width=1024,
        height=1024,
        sampler="Euler a",
        base_model_id="anything-v5",
        base_model_architecture="SD1.5",
        loras_used=[
            {"name": "anime-style-lora", "weight": 0.9, "source": "civitai"},
            {"name": "detail-tweaker", "weight": 0.5, "source": "local"},
        ],
        generation_time_seconds=12.8,
        peak_vram_gb=6.5,
    )

    dummy_image = Image.new("RGB", (1024, 1024), color=(255, 192, 203))
    writer = ImageMetadataWriter()

    saved_path = writer.save_with_metadata(
        image=dummy_image,
        metadata=metadata,
        output_dir=Path("./outputs"),
    )

    print(f"✓ Image saved: {saved_path.name}")

    # Now extract metadata from the saved image
    extracted = writer.extract_metadata(saved_path)

    if extracted:
        print(f"\n✓ Metadata extracted successfully:")
        print(f"  - Generation ID: {extracted.generation_id}")
        print(f"  - Timestamp: {extracted.generation_timestamp}")
        print(f"  - Prompt: {extracted.prompt}")
        print(f"  - Seed: {extracted.seed}")
        print(f"  - Steps: {extracted.steps}")
        print(f"  - CFG Scale: {extracted.cfg_scale}")
        print(f"  - Resolution: {extracted.width}x{extracted.height}")
        print(f"  - Sampler: {extracted.sampler}")
        print(f"  - Base Model: {extracted.base_model_id}")
        print(f"  - Architecture: {extracted.base_model_architecture}")
        print(f"  - LoRAs used: {len(extracted.loras_used)}")
        for lora in extracted.loras_used:
            print(f"    - {lora['name']} (weight: {lora['weight']})")
        print(f"  - Generation time: {extracted.generation_time_seconds}s")
        print(f"  - Peak VRAM: {extracted.peak_vram_gb}GB")

        print(f"\n✓ Full reproduction possible:")
        print(f"  All parameters needed to reproduce this image are embedded!")
    else:
        print("✗ Failed to extract metadata")


def example_6_sidecar_json():
    """Example 6: Saving metadata as sidecar JSON file."""
    print("\n=== Example 6: Sidecar JSON Files ===\n")

    metadata = ImageMetadataEmbedding(
        generation_id=create_generation_id(),
        generation_timestamp=create_timestamp(),
        prompt="realistic portrait photography",
        negative_prompt="cartoon, anime, illustration",
        seed=555,
        steps=50,
        cfg_scale=8.5,
        width=768,
        height=1024,
        sampler="DPM++ 2M SDE",
        base_model_id="realistic-vision-v5",
        base_model_architecture="SD1.5",
    )

    dummy_image = Image.new("RGB", (768, 1024), color=(128, 128, 128))
    writer = ImageMetadataWriter()

    # Save with sidecar JSON
    output_path = writer.save_with_metadata(
        image=dummy_image,
        metadata=metadata,
        output_dir=Path("./outputs"),
        save_sidecar_json=True,  # Also save as .metadata.json
    )

    sidecar_path = output_path.with_suffix(".metadata.json")

    print(f"✓ Image saved: {output_path.name}")
    print(f"✓ Sidecar JSON saved: {sidecar_path.name}")
    print(f"\n✓ Benefits of sidecar JSON:")
    print(f"  - Easy to read/edit with text editor")
    print(f"  - Can be indexed by search tools")
    print(f"  - Survives image format conversions")
    print(f"  - Metadata embedded in image AND separate file")


def example_7_privacy_security():
    """Example 7: Privacy and security features."""
    print("\n=== Example 7: Privacy & Security ===\n")

    print("✓ Privacy-focused design:")
    print("  - No user tracking or analytics")
    print("  - No API keys stored in metadata")
    print("  - No personally identifiable information")
    print("  - GUID-only naming available for anonymity")
    print("  - All metadata is local to the image")
    print()
    print("✓ Security features:")
    print("  - Metadata is read-only after embedding")
    print("  - No code execution in metadata")
    print("  - Standard PNG/EXIF formats only")
    print("  - Sanitized filenames (no special chars)")
    print()
    print("✓ What's included in metadata:")
    print("  - Generation parameters (reproducibility)")
    print("  - Model information (tracking)")
    print("  - Performance metrics (optimization)")
    print("  - Timestamp and GUID (uniqueness)")
    print()
    print("✓ What's NOT included:")
    print("  - User identity or personal info")
    print("  - API keys or credentials")
    print("  - System paths or environment details")
    print("  - Network or location data")


def example_8_integration_with_generation_result():
    """Example 8: Integration with GenerationResult."""
    print("\n=== Example 8: Integration with GenerationResult ===\n")

    print("The metadata system is integrated into GenerationResult:")
    print()
    print("# Method 1: Auto-generated naming")
    print("result.save('/outputs', use_auto_naming=True)")
    print("# Result: /outputs/20250111_143022_a3f2e9d4.png")
    print()
    print("# Method 2: Descriptive naming")
    print("result.save(")
    print("    '/outputs',")
    print("    use_auto_naming=True,")
    print("    naming_config=ImageNamingConfig.descriptive()")
    print(")")
    print("# Result: /outputs/20250111_143022_beautiful-sunset_a3f2e9d4.png")
    print()
    print("# Method 3: Custom filename")
    print("result.save('/outputs/my_custom_name.png')")
    print("# Result: /outputs/my_custom_name.png")
    print()
    print("All methods automatically embed full metadata in the image!")


if __name__ == "__main__":
    print("=" * 60)
    print("Image Metadata & Naming System - Examples")
    print("=" * 60)

    # Create outputs directory
    Path("./outputs").mkdir(exist_ok=True)

    # Run examples
    example_1_basic_metadata()
    example_2_descriptive_naming()
    example_3_guid_only()
    example_4_custom_naming()
    example_5_metadata_extraction()
    example_6_sidecar_json()
    example_7_privacy_security()
    example_8_integration_with_generation_result()

    print("\n" + "=" * 60)
    print("✓ All examples completed!")
    print("=" * 60)
    print(f"\nCheck the ./outputs directory for generated images.")
    print("Each image contains embedded metadata that can be extracted.")
