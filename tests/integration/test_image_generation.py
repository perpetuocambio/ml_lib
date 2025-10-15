"""
Integration tests for image generation pipeline.

These tests verify the complete image generation flow with real models.
"""

import pytest
from pathlib import Path
from PIL import Image

from ml_lib.diffusion.generation.facade import ImageGenerator, GenerationOptions
from ml_lib.diffusion.services import IntelligentPipelineBuilder
from ml_lib.diffusion.generation.pipeline import PipelineConfig


@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory for test images."""
    output = tmp_path / "test_outputs"
    output.mkdir()
    return output


@pytest.fixture
def simple_generator():
    """Create a simple image generator for testing."""
    options = GenerationOptions(
        steps=20,  # Reduced for faster testing
        width=512,
        height=512,
        enable_loras=False,  # Disable LoRAs for basic test
        enable_learning=False,
    )
    return ImageGenerator(
        model="stabilityai/stable-diffusion-2-1-base", device="cuda", options=options
    )


class TestBasicGeneration:
    """Test basic image generation functionality."""

    def test_simple_generation(self, simple_generator, output_dir):
        """Test generating a simple image."""
        print("\n🎨 Testing basic image generation...")

        image = simple_generator.generate_from_prompt(
            prompt="a beautiful sunset over mountains", seed=42
        )

        # Verify image properties
        assert isinstance(image, Image.Image)
        assert image.size == (512, 512)

        # Save for manual inspection
        output_path = output_dir / "test_simple_generation.png"
        image.save(output_path)
        print(f"✅ Image saved to: {output_path}")

    def test_generation_with_negative_prompt(self, simple_generator, output_dir):
        """Test generation with negative prompt."""
        print("\n🎨 Testing generation with negative prompt...")

        image = simple_generator.generate_from_prompt(
            prompt="a beautiful woman with blue eyes",
            negative_prompt="ugly, deformed, low quality, blurry",
            seed=42,
        )

        assert isinstance(image, Image.Image)

        output_path = output_dir / "test_negative_prompt.png"
        image.save(output_path)
        print(f"✅ Image saved to: {output_path}")

    def test_generation_with_custom_params(self, simple_generator, output_dir):
        """Test generation with custom parameters."""
        print("\n🎨 Testing generation with custom parameters...")

        image = simple_generator.generate_from_prompt(
            prompt="a cyberpunk city at night, neon lights",
            steps=15,
            cfg_scale=9.0,
            width=768,
            height=512,
            seed=123,
        )

        assert isinstance(image, Image.Image)
        assert image.size == (768, 512)

        output_path = output_dir / "test_custom_params.png"
        image.save(output_path)
        print(f"✅ Image saved to: {output_path}")


class TestPromptAnalysis:
    """Test prompt analysis functionality."""

    def test_analyze_prompt(self, simple_generator):
        """Test prompt analysis without generation."""
        print("\n🔍 Testing prompt analysis...")

        analysis = simple_generator.analyze_prompt(
            "anime girl with magical powers, fantasy art"
        )

        # Verify analysis structure
        assert analysis is not None
        assert hasattr(analysis, "concepts")
        assert hasattr(analysis, "emphases")
        assert hasattr(analysis, "reasoning")

        print(f"✅ Concepts detected: {analysis.concept_count}")
        print(f"✅ Emphases found: {analysis.emphasis_count}")


class TestIntelligentPipeline:
    """Test intelligent pipeline with Ollama integration."""

    @pytest.fixture
    def intelligent_pipeline(self):
        """Create intelligent pipeline with Ollama using dolphin3 (NSFW-capable)."""
        from ml_lib.diffusion.generation.pipeline import OllamaConfig

        config = PipelineConfig(
            base_model="stabilityai/stable-diffusion-2-1-base",
            ollama_config=OllamaConfig(
                base_url="http://localhost:11434",
                model="dolphin3",  # NSFW-capable model
                enabled=True,
            ),
            enable_learning=False,
        )

        return IntelligentPipelineBuilder.from_diffusers(config)

    def test_intelligent_generation_with_ollama(self, intelligent_pipeline, output_dir):
        """Test generation with Ollama semantic analysis."""
        print("\n🤖 Testing intelligent generation with Ollama...")

        result = intelligent_pipeline.generate(
            prompt="a portrait of a wise old wizard with a long white beard", seed=42
        )

        # Verify result structure
        assert result is not None
        assert isinstance(result.image, Image.Image)
        assert result.metadata is not None
        assert result.explanation is not None

        # Save image
        output_path = output_dir / "test_ollama_generation.png"
        result.image.save(output_path)

        print(f"✅ Image saved to: {output_path}")
        print(f"📝 Explanation: {result.explanation.summary}")
        print(f"⏱️  Generation time: {result.metadata.generation_time_seconds:.2f}s")
        print(f"💾 Peak VRAM: {result.metadata.peak_vram_gb:.2f}GB")

    def test_analyze_and_recommend(self, intelligent_pipeline):
        """Test analyze and recommend workflow."""
        print("\n🔍 Testing analyze and recommend...")

        recommendations = intelligent_pipeline.analyze_and_recommend(
            prompt="cyberpunk city at night with neon lights"
        )

        assert recommendations is not None
        assert recommendations.prompt_analysis is not None
        assert recommendations.suggested_params is not None

        print(f"✅ Recommendations generated")
        print(f"📊 Suggested steps: {recommendations.suggested_params.num_steps}")
        print(f"📊 Suggested CFG: {recommendations.suggested_params.guidance_scale}")
        print(
            f"📊 Suggested resolution: {recommendations.suggested_params.width}x{recommendations.suggested_params.height}"
        )


class TestMemoryOptimization:
    """Test memory optimization features."""

    @pytest.mark.parametrize("memory_mode", ["balanced", "aggressive"])
    def test_memory_modes(self, memory_mode, output_dir):
        """Test different memory optimization modes."""
        print(f"\n💾 Testing memory mode: {memory_mode}")

        options = GenerationOptions(
            steps=15,
            width=512,
            height=512,
            memory_mode=memory_mode,
            enable_loras=False,
        )

        generator = ImageGenerator(
            model="stabilityai/stable-diffusion-2-1-base", options=options
        )

        image = generator.generate_from_prompt(prompt="a simple landscape", seed=42)

        assert isinstance(image, Image.Image)

        output_path = output_dir / f"test_memory_{memory_mode}.png"
        image.save(output_path)
        print(f"✅ Image saved to: {output_path}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
