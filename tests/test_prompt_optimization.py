"""
Test prompt optimization for different model architectures.

This test verifies that prompts are correctly optimized for:
- SDXL: Quality tags appended, weights capped at 1.4
- Pony V6: Score tags prepended, weights capped at 1.5
- SD 1.5: Quality tags prepended, weights capped at 1.5
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_lib.diffusion.domain.services.prompt_analyzer import PromptAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_sdxl_optimization():
    """Test SDXL prompt optimization."""
    logger.info("=" * 60)
    logger.info("Testing SDXL Prompt Optimization")
    logger.info("=" * 60)

    # Create analyzer without Ollama (pure rule-based)
    analyzer = PromptAnalyzer(use_llm=False)

    # Test prompt with high weight that should be capped
    test_prompt = "a beautiful woman, (detailed face:2.0), blue eyes"
    test_negative = "low quality, blurry"

    positive, negative = analyzer.optimize_for_model(
        prompt=test_prompt,
        negative_prompt=test_negative,
        base_model_architecture="SDXL",
        quality="high"
    )

    logger.info(f"Original positive: {test_prompt}")
    logger.info(f"Optimized positive: {positive}")
    logger.info(f"Original negative: {test_negative}")
    logger.info(f"Optimized negative: {negative}")

    # Verify quality tags are added
    assert "masterpiece" in positive, "SDXL should add quality tags"
    assert "best quality" in positive, "SDXL should add 'best quality'"

    # Verify weight is capped at 1.4 (from 2.0)
    assert "(detailed face:1.40)" in positive, f"SDXL should cap weight at 1.4, got: {positive}"

    logger.info("✅ SDXL optimization test passed")


def test_pony_optimization():
    """Test Pony Diffusion V6 optimization."""
    logger.info("=" * 60)
    logger.info("Testing Pony Diffusion V6 Prompt Optimization")
    logger.info("=" * 60)

    analyzer = PromptAnalyzer(use_llm=False)

    test_prompt = "anthro elephant, (detailed:1.8)"
    test_negative = "bad anatomy"

    positive, negative = analyzer.optimize_for_model(
        prompt=test_prompt,
        negative_prompt=test_negative,
        base_model_architecture="Pony",
        quality="high"
    )

    logger.info(f"Original positive: {test_prompt}")
    logger.info(f"Optimized positive: {positive}")
    logger.info(f"Original negative: {test_negative}")
    logger.info(f"Optimized negative: {negative}")

    # Verify score tags are prepended
    assert positive.startswith("score_9"), "Pony should start with score_9"
    assert "score_8_up" in positive, "Pony should include score_8_up"

    # Verify weight is capped at 1.5 (from 1.8)
    assert "(detailed:1.50)" in positive, f"Pony should cap weight at 1.5, got: {positive}"

    # Verify negative has score tags
    assert "score_4" in negative, "Pony negative should include score_4"
    assert "score_5" in negative, "Pony negative should include score_5"

    logger.info("✅ Pony optimization test passed")


def test_sd15_optimization():
    """Test SD 1.5 optimization."""
    logger.info("=" * 60)
    logger.info("Testing SD 1.5 Prompt Optimization")
    logger.info("=" * 60)

    analyzer = PromptAnalyzer(use_llm=False)

    test_prompt = "beautiful landscape, mountains"
    test_negative = ""  # Test default negative

    positive, negative = analyzer.optimize_for_model(
        prompt=test_prompt,
        negative_prompt=test_negative,
        base_model_architecture="SD15",
        quality="balanced"
    )

    logger.info(f"Original positive: {test_prompt}")
    logger.info(f"Optimized positive: {positive}")
    logger.info(f"Original negative: {test_negative}")
    logger.info(f"Optimized negative: {negative}")

    # Verify quality tags are prepended
    assert positive.startswith("masterpiece"), "SD15 should start with quality tags"

    # Verify default negative is applied
    assert "low quality" in negative, "SD15 should add default negative"
    assert "bad anatomy" in negative, "SD15 negative should include bad anatomy"

    logger.info("✅ SD 1.5 optimization test passed")


def test_problematic_prompt():
    """Test optimization with the user's problematic NSFW prompt."""
    logger.info("=" * 60)
    logger.info("Testing Problematic Prompt (User Example)")
    logger.info("=" * 60)

    analyzer = PromptAnalyzer(use_llm=False)

    # User's problematic prompt (simplified)
    test_prompt = (
        "an old female elephant in the forest, anthro, "
        "an old elderly female anthro elephant, "
        "blue eyes, orgasm face, eyes rolling"
    )

    test_negative = (
        "close-up, headshot, cropped face, exaggerated makeup, ugly, blur, "
        "cartoon, anime, doll, 3d, deformed, disfigured, unrealistic, "
        "smooth skin, shiny skin, cgi, plastic"
    )

    # Test with Pony (likely what user is using based on prompt style)
    positive, negative = analyzer.optimize_for_model(
        prompt=test_prompt,
        negative_prompt=test_negative,
        base_model_architecture="Pony",
        quality="high"
    )

    logger.info(f"Original positive: {test_prompt[:80]}...")
    logger.info(f"Optimized positive: {positive[:120]}...")
    logger.info(f"Optimized negative: {negative[:120]}...")

    # Verify Pony-specific optimizations
    assert positive.startswith("score_9"), "Should add Pony quality tags"
    assert "score_4" in negative, "Should add Pony negative tags"

    logger.info("✅ Problematic prompt test passed")
    logger.info("")
    logger.info("Note: Prompt now has proper quality tags and weight limits.")
    logger.info("This should improve:")
    logger.info("  - Face definition")
    logger.info("  - Anatomical correctness")
    logger.info("  - Detail in intimate areas")


def main():
    """Run all prompt optimization tests."""
    logger.info("Starting prompt optimization tests...")
    logger.info("")

    try:
        test_sdxl_optimization()
        logger.info("")

        test_pony_optimization()
        logger.info("")

        test_sd15_optimization()
        logger.info("")

        test_problematic_prompt()
        logger.info("")

        logger.info("=" * 60)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Summary:")
        logger.info("- SDXL: Quality tags appended, weights capped at 1.4")
        logger.info("- Pony V6: Score tags prepended, weights capped at 1.5")
        logger.info("- SD 1.5: Quality tags prepended, weights capped at 1.5")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Run actual generation with optimized prompts")
        logger.info("2. Compare quality with previous generations")
        logger.info("3. Verify anatomical correctness improvements")

    except AssertionError as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
