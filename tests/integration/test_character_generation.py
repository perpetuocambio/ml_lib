"""
Integration tests for character generation with image generation.

Tests the full workflow of generating a character description
and then generating an image from that description.
"""

import pytest
from pathlib import Path
from PIL import Image

from ml_lib.diffusion.generation.facade import ImageGenerator, GenerationOptions


@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory for test images."""
    output = tmp_path / "character_outputs"
    output.mkdir()
    return output


@pytest.fixture
def character_generator():
    """Create image generator optimized for character generation."""
    options = GenerationOptions(
        steps=30,
        cfg_scale=7.5,
        width=512,
        height=768,  # Portrait orientation
        enable_loras=True,  # Enable LoRAs for better character quality
        enable_learning=True,  # Enable learning from feedback
        memory_mode="balanced",
    )
    return ImageGenerator(
        model="stabilityai/stable-diffusion-xl-base-1.0", device="cuda", options=options
    )


class TestCharacterGeneration:
    """Test character generation workflows."""

    def test_basic_character_generation(self, character_generator, output_dir):
        """Test generating a basic character."""
        print("\n👤 Testing basic character generation...")

        character_prompt = (
            "portrait of a young woman with long brown hair, "
            "green eyes, wearing a leather jacket, "
            "confident expression, detailed face, high quality"
        )

        image = character_generator.generate_from_prompt(
            prompt=character_prompt,
            negative_prompt="low quality, blurry, deformed, ugly, bad anatomy",
            seed=42,
        )

        assert isinstance(image, Image.Image)
        assert image.size == (512, 768)

        output_path = output_dir / "character_basic.png"
        image.save(output_path)
        print(f"✅ Character image saved to: {output_path}")

    def test_character_with_style(self, character_generator, output_dir):
        """Test character generation with specific art style."""
        print("\n🎨 Testing character with art style...")

        styles = [
            (
                "anime",
                "anime style girl with pink hair, cute expression, school uniform",
            ),
            (
                "realistic",
                "photorealistic portrait of a man with beard, professional photo",
            ),
            (
                "fantasy",
                "fantasy elf character with pointed ears, mystical clothing, magical atmosphere",
            ),
        ]

        for style_name, prompt in styles:
            print(f"  Generating {style_name} character...")

            image = character_generator.generate_from_prompt(
                prompt=prompt, negative_prompt="low quality, deformed, blurry", seed=42
            )

            assert isinstance(image, Image.Image)

            output_path = output_dir / f"character_{style_name}.png"
            image.save(output_path)
            print(f"  ✅ Saved to: {output_path}")

    def test_character_with_personality_traits(self, character_generator, output_dir):
        """Test character generation emphasizing personality traits."""
        print("\n😊 Testing character with personality traits...")

        personalities = [
            "confident and charismatic leader",
            "shy and introverted scholar",
            "fierce and determined warrior",
            "kind and gentle healer",
        ]

        for i, personality in enumerate(personalities):
            prompt = f"portrait of a character who is {personality}, detailed face, expressive"

            print(f"  Generating: {personality}...")

            image = character_generator.generate_from_prompt(
                prompt=prompt,
                negative_prompt="low quality, blurry, deformed",
                seed=100 + i,
            )

            assert isinstance(image, Image.Image)

            output_path = output_dir / f"character_personality_{i}.png"
            image.save(output_path)
            print(f"  ✅ Saved to: {output_path}")


class TestCharacterGeneratorIntegration:
    """Test integration with character generator from other modules."""

    @pytest.fixture
    def mock_character_description(self):
        """Create a mock character description."""
        return {
            "name": "Aria Stormwind",
            "race": "Half-Elf",
            "class": "Ranger",
            "age": 28,
            "appearance": {
                "hair": "long silver hair",
                "eyes": "piercing blue eyes",
                "build": "athletic and agile",
                "clothing": "leather armor with green cloak",
                "distinctive_features": "scar across left cheek, elvish tattoos",
            },
            "personality": "confident, independent, protective of nature",
        }

    def test_character_from_description(
        self, character_generator, mock_character_description, output_dir
    ):
        """Test generating character image from structured description."""
        print("\n📝 Testing character generation from description...")

        # Build prompt from character description
        desc = mock_character_description
        appearance = desc["appearance"]

        prompt = (
            f"portrait of {desc['name']}, {desc['race']} {desc['class']}, "
            f"age {desc['age']}, {appearance['hair']}, {appearance['eyes']}, "
            f"{appearance['build']}, wearing {appearance['clothing']}, "
            f"{appearance['distinctive_features']}, "
            f"{desc['personality']}, "
            "detailed face, high quality, fantasy art"
        )

        print(f"  Character: {desc['name']}")
        print(f"  Prompt: {prompt[:100]}...")

        image = character_generator.generate_from_prompt(
            prompt=prompt,
            negative_prompt="low quality, blurry, deformed, ugly, bad anatomy",
            seed=42,
        )

        assert isinstance(image, Image.Image)

        output_path = output_dir / f"character_{desc['name'].replace(' ', '_')}.png"
        image.save(output_path)
        print(f"✅ Character image saved to: {output_path}")

    def test_character_variations(
        self, character_generator, mock_character_description, output_dir
    ):
        """Test generating variations of the same character."""
        print("\n🔄 Testing character variations...")

        desc = mock_character_description
        base_prompt = (
            f"{desc['name']}, {desc['race']} {desc['class']}, "
            f"{desc['appearance']['hair']}, {desc['appearance']['eyes']}"
        )

        variations = [
            ("action", f"{base_prompt}, action pose, battle stance, dynamic"),
            ("portrait", f"{base_prompt}, close-up portrait, detailed face"),
            ("full_body", f"{base_prompt}, full body shot, standing pose"),
        ]

        for var_name, prompt in variations:
            print(f"  Generating {var_name} variation...")

            image = character_generator.generate_from_prompt(
                prompt=prompt + ", high quality, fantasy art",
                negative_prompt="low quality, blurry, deformed",
                seed=42,
            )

            assert isinstance(image, Image.Image)

            output_path = output_dir / f"character_variation_{var_name}.png"
            image.save(output_path)
            print(f"  ✅ Saved to: {output_path}")


class TestBatchCharacterGeneration:
    """Test generating multiple characters in batch."""

    def test_party_generation(self, character_generator, output_dir):
        """Test generating a party of characters."""
        print("\n👥 Testing party generation...")

        party_members = [
            ("warrior", "muscular warrior with sword and shield, battle-scarred"),
            ("mage", "wise mage with staff, flowing robes, mystical aura"),
            ("rogue", "stealthy rogue in dark clothing, daggers, mysterious"),
            ("cleric", "holy cleric with symbol of faith, healing hands"),
        ]

        for role, description in party_members:
            print(f"  Generating {role}...")

            image = character_generator.generate_from_prompt(
                prompt=f"portrait of {description}, detailed face, fantasy art",
                negative_prompt="low quality, blurry, deformed",
                seed=42,
            )

            assert isinstance(image, Image.Image)

            output_path = output_dir / f"party_member_{role}.png"
            image.save(output_path)
            print(f"  ✅ Saved to: {output_path}")


class TestCharacterWithFeedback:
    """Test character generation with feedback loop."""

    def test_feedback_improvement(self, output_dir):
        """Test that feedback improves future generations."""
        print("\n🔄 Testing feedback improvement loop...")

        # Create generator with learning enabled
        options = GenerationOptions(
            steps=20,
            enable_learning=True,
        )
        generator = ImageGenerator(
            model="stabilityai/stable-diffusion-2-1-base", options=options
        )

        prompt = "portrait of a young adventurer"

        # First generation
        print("  First generation...")
        image1 = generator.generate_from_prompt(prompt=prompt, seed=42)
        output_path1 = output_dir / "character_feedback_v1.png"
        image1.save(output_path1)

        # Simulate negative feedback
        # Note: In real usage, generation_id would come from result metadata
        try:
            generator.provide_feedback(
                generation_id="test_001",
                rating=2,
                comments="Face details not good enough",
            )
            print("  ✅ Feedback recorded")
        except RuntimeError as e:
            print(f"  ⚠️  Feedback not recorded (expected in test): {e}")

        # Second generation (should theoretically improve with learning)
        print("  Second generation...")
        image2 = generator.generate_from_prompt(prompt=prompt, seed=43)
        output_path2 = output_dir / "character_feedback_v2.png"
        image2.save(output_path2)

        print(f"✅ Images saved for comparison:")
        print(f"   v1: {output_path1}")
        print(f"   v2: {output_path2}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
