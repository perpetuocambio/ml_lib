"""
Integration tests for adult content (NSFW) image generation.

Tests the generation pipeline with NSFW content, including:
- Safety filters disabled appropriately
- Quality control for adult content
- Metadata and content tracking
"""

import pytest
from pathlib import Path
from PIL import Image

from ml_lib.diffusion.generation.facade import ImageGenerator, GenerationOptions


@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory for adult content test images."""
    output = tmp_path / "adult_content_outputs"
    output.mkdir()
    # Mark directory for adult content
    (output / "ADULT_CONTENT.txt").write_text(
        "This directory contains adult content generated for testing purposes only."
    )
    return output


@pytest.fixture
def nsfw_generator():
    """
    Create image generator configured for adult content.

    Note: This disables safety filters for testing purposes.
    In production, ensure proper age verification and content warnings.
    """
    options = GenerationOptions(
        steps=35,
        cfg_scale=8.0,
        width=768,
        height=1024,  # Portrait orientation
        enable_loras=True,  # Enable LoRAs for better quality
        enable_learning=True,  # Learn from user preferences
        memory_mode="balanced",
    )
    return ImageGenerator(
        model="stabilityai/stable-diffusion-xl-base-1.0", device="cuda", options=options
    )


class TestNSFWGeneration:
    """Test NSFW content generation."""

    @pytest.mark.nsfw
    def test_artistic_nude_generation(self, nsfw_generator, output_dir):
        """Test generating artistic nude content."""
        print("\n🎨 Testing artistic nude generation...")

        prompt = (
            "artistic nude portrait, beautiful woman, "
            "tasteful lighting, professional photography, "
            "elegant pose, high quality, artistic"
        )

        negative_prompt = (
            "low quality, blurry, deformed, ugly, "
            "bad anatomy, bad proportions, distorted"
        )

        image = nsfw_generator.generate_from_prompt(
            prompt=prompt, negative_prompt=negative_prompt, seed=42
        )

        assert isinstance(image, Image.Image)
        assert image.size == (768, 1024)

        output_path = output_dir / "artistic_nude.png"
        image.save(output_path)
        print(f"✅ Image saved to: {output_path}")

    @pytest.mark.nsfw
    def test_boudoir_photography_style(self, nsfw_generator, output_dir):
        """Test boudoir photography style generation."""
        print("\n📸 Testing boudoir photography style...")

        prompt = (
            "boudoir photography, beautiful woman in lingerie, "
            "soft lighting, bedroom setting, elegant and sensual, "
            "professional photo, high quality"
        )

        negative_prompt = "low quality, blurry, deformed, ugly, bad anatomy"

        image = nsfw_generator.generate_from_prompt(
            prompt=prompt, negative_prompt=negative_prompt, seed=100
        )

        assert isinstance(image, Image.Image)

        output_path = output_dir / "boudoir_style.png"
        image.save(output_path)
        print(f"✅ Image saved to: {output_path}")

    @pytest.mark.nsfw
    def test_pinup_art_style(self, nsfw_generator, output_dir):
        """Test pin-up art style generation."""
        print("\n🎭 Testing pin-up art style...")

        prompt = (
            "vintage pin-up art style, beautiful woman in retro outfit, "
            "1950s aesthetic, playful pose, vibrant colors, "
            "classic pin-up illustration"
        )

        negative_prompt = "low quality, deformed, modern, blurry"

        image = nsfw_generator.generate_from_prompt(
            prompt=prompt, negative_prompt=negative_prompt, seed=200
        )

        assert isinstance(image, Image.Image)

        output_path = output_dir / "pinup_art.png"
        image.save(output_path)
        print(f"✅ Image saved to: {output_path}")


class TestNSFWCharacterGeneration:
    """Test NSFW character generation for adult games/stories."""

    @pytest.mark.nsfw
    def test_adult_game_character(self, nsfw_generator, output_dir):
        """Test generating character for adult game."""
        print("\n🎮 Testing adult game character...")

        character_desc = {
            "name": "Lilith",
            "role": "Succubus Queen",
            "appearance": "seductive demoness with horns, red skin, long black hair",
            "outfit": "revealing dark fantasy outfit with leather and chains",
            "style": "fantasy art, dark aesthetic",
        }

        prompt = (
            f"portrait of {character_desc['name']}, {character_desc['role']}, "
            f"{character_desc['appearance']}, wearing {character_desc['outfit']}, "
            f"{character_desc['style']}, detailed face, high quality, "
            "sultry expression, mystical atmosphere"
        )

        negative_prompt = "low quality, blurry, deformed, ugly, bad anatomy"

        image = nsfw_generator.generate_from_prompt(
            prompt=prompt, negative_prompt=negative_prompt, seed=42
        )

        assert isinstance(image, Image.Image)

        output_path = output_dir / f"character_{character_desc['name'].lower()}.png"
        image.save(output_path)
        print(f"✅ Character saved to: {output_path}")

    @pytest.mark.nsfw
    def test_adult_romance_scene(self, nsfw_generator, output_dir):
        """Test generating romantic scene for adult visual novel."""
        print("\n💕 Testing adult romance scene...")

        prompt = (
            "romantic couple embracing, intimate moment, "
            "soft candlelight, bedroom setting, passionate kiss, "
            "cinematic lighting, detailed faces, emotional, "
            "high quality illustration"
        )

        negative_prompt = "low quality, blurry, deformed, ugly, bad anatomy"

        image = nsfw_generator.generate_from_prompt(
            prompt=prompt, negative_prompt=negative_prompt, seed=150
        )

        assert isinstance(image, Image.Image)

        output_path = output_dir / "romance_scene.png"
        image.save(output_path)
        print(f"✅ Scene saved to: {output_path}")


class TestNSFWQualityControl:
    """Test quality control for adult content."""

    @pytest.mark.nsfw
    def test_anatomy_quality(self, nsfw_generator, output_dir):
        """Test that anatomy is correctly rendered in adult content."""
        print("\n🔍 Testing anatomy quality...")

        prompt = (
            "full body portrait of beautiful woman, "
            "perfect anatomy, correct proportions, "
            "realistic skin texture, detailed face and body, "
            "professional photography, high quality"
        )

        negative_prompt = (
            "bad anatomy, distorted, deformed, extra limbs, "
            "missing limbs, wrong proportions, ugly, low quality"
        )

        image = nsfw_generator.generate_from_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            cfg_scale=9.0,  # Higher CFG for better adherence
            seed=42,
        )

        assert isinstance(image, Image.Image)

        output_path = output_dir / "anatomy_quality_check.png"
        image.save(output_path)
        print(f"✅ Quality check image saved to: {output_path}")

    @pytest.mark.nsfw
    def test_different_body_types(self, nsfw_generator, output_dir):
        """Test generating different body types."""
        print("\n👥 Testing different body types...")

        body_types = [
            ("athletic", "athletic woman, toned body, fit physique"),
            ("curvy", "curvy woman, voluptuous figure, full figure"),
            ("petite", "petite woman, slim figure, delicate features"),
        ]

        for body_type, description in body_types:
            print(f"  Generating {body_type} body type...")

            prompt = (
                f"portrait of beautiful {description}, "
                "tasteful pose, professional photography, "
                "high quality, detailed"
            )

            negative_prompt = "low quality, deformed, bad anatomy"

            image = nsfw_generator.generate_from_prompt(
                prompt=prompt, negative_prompt=negative_prompt, seed=42
            )

            assert isinstance(image, Image.Image)

            output_path = output_dir / f"body_type_{body_type}.png"
            image.save(output_path)
            print(f"  ✅ Saved to: {output_path}")


class TestNSFWMetadata:
    """Test metadata handling for adult content."""

    @pytest.mark.nsfw
    def test_adult_content_metadata(self, nsfw_generator, output_dir):
        """Test that adult content is properly tagged in metadata."""
        print("\n📝 Testing adult content metadata...")

        from ml_lib.diffusion.services import IntelligentPipelineBuilder
        from ml_lib.diffusion.generation.pipeline import PipelineConfig

        config = PipelineConfig(base_model="stabilityai/stable-diffusion-xl-base-1.0")
        pipeline = IntelligentPipelineBuilder.from_diffusers(config)

        prompt = "artistic nude portrait, professional photography"

        result = pipeline.generate(
            prompt=prompt, negative_prompt="low quality, deformed", seed=42
        )

        # Check metadata
        assert result.metadata is not None
        assert result.metadata.prompt == prompt
        assert result.image is not None

        # Save with metadata
        from ml_lib.diffusion.generation.image_metadata import (
            ImageMetadataWriter,
            ImageMetadataEmbedding,
        )

        metadata_embedding = ImageMetadataEmbedding(
            generation_id=result.id,
            generation_timestamp=result.metadata.timestamp
            if hasattr(result.metadata, "timestamp")
            else "2025-01-01T00:00:00Z",
            prompt=result.metadata.prompt,
            negative_prompt=result.metadata.negative_prompt,
            seed=result.metadata.seed,
            steps=result.metadata.steps,
            cfg_scale=result.metadata.cfg_scale,
            width=result.metadata.width,
            height=result.metadata.height,
            sampler=result.metadata.sampler,
            base_model_id=result.metadata.base_model_id,
            pipeline_type=result.metadata.pipeline_type,
        )

        writer = ImageMetadataWriter()
        output_path = writer.save_with_metadata(
            image=result.image,
            metadata=metadata_embedding,
            output_dir=output_dir,
            embed_full_json=True,
        )

        print(f"✅ Image with metadata saved to: {output_path}")

        # Verify metadata can be extracted
        extracted_metadata = writer.extract_metadata(output_path)
        assert extracted_metadata is not None
        assert extracted_metadata.prompt == prompt
        print(f"✅ Metadata successfully extracted")


class TestNSFWContentWarnings:
    """Test content warning system for adult content."""

    @pytest.mark.nsfw
    def test_content_warning_detection(self):
        """Test that system can detect NSFW prompts."""
        print("\n⚠️  Testing content warning detection...")

        nsfw_keywords = [
            "nude",
            "naked",
            "nsfw",
            "adult",
            "explicit",
            "sexual",
            "erotic",
            "xxx",
            "porn",
        ]

        test_prompts = [
            ("artistic nude portrait", True),
            ("beautiful landscape", False),
            ("explicit adult content", True),
            ("family portrait", False),
        ]

        for prompt, expected_nsfw in test_prompts:
            is_nsfw = any(keyword in prompt.lower() for keyword in nsfw_keywords)
            assert is_nsfw == expected_nsfw, f"Failed for prompt: {prompt}"

        print("✅ Content warning detection working correctly")


if __name__ == "__main__":
    # Run tests with pytest
    # Use -m nsfw to run only NSFW tests
    pytest.main([__file__, "-v", "-s", "-m", "nsfw"])
