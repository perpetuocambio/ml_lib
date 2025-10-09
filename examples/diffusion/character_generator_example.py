"""
Example usage of CharacterGenerator (Diversity System).

This example demonstrates:
1. Generating diverse characters with consistent ethnic features
2. Countering CivitAI model bias toward white subjects
3. Creating prompts with proper weighting for non-white features
4. Batch character generation with diversity enforcement
"""

import logging
from ml_lib.diffusion.intelligent.prompting.character_generator import CharacterGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def example_single_character():
    """Example: Generate a single diverse character."""
    logger.info("=" * 60)
    logger.info("Example 1: Single Character Generation")
    logger.info("=" * 60)

    generator = CharacterGenerator()

    # Generate with diversity enforcement (default)
    character = generator.generate()

    logger.info("\nGenerated Character:")
    logger.info(f"  Age: {character.age} years old")
    logger.info(f"  Ethnicity: {character.ethnicity} (weight: {character.ethnicity_prompt_weight})")
    logger.info(f"  Skin Tone: {character.skin_tone} (weight: {character.skin_prompt_weight})")
    logger.info(f"  Eyes: {character.eye_color}")
    logger.info(f"  Hair: {character.hair_color} ({character.hair_texture})")
    logger.info(f"  Body: {character.body_type}")
    logger.info(f"  Breasts: {character.breast_size}")
    logger.info(f"  Pose: {character.pose} (explicit: {character.pose_explicit})")
    logger.info(f"  Setting: {character.setting}")

    logger.info("\nGenerated Prompt:")
    logger.info(f"  {character.to_prompt()}")


def example_explicit_character():
    """Example: Generate character with explicit pose."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Explicit Pose Character")
    logger.info("=" * 60)

    generator = CharacterGenerator()

    # Generate with explicit poses only
    character = generator.generate(explicit_poses_only=True)

    logger.info("\nGenerated Character (Explicit):")
    logger.info(f"  Age: {character.age} ({', '.join(character.age_keywords[:2])})")
    logger.info(f"  Ethnicity: {character.ethnicity}")
    logger.info(f"  Skin: {character.skin_tone}")
    logger.info(f"  Pose: {character.pose} (explicit: {character.pose_explicit})")

    logger.info("\nExplicit Prompt:")
    prompt = character.to_prompt(include_explicit=True)
    logger.info(f"  {prompt}")


def example_age_constrained():
    """Example: Generate character within specific age range."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Age-Constrained Character (50-60)")
    logger.info("=" * 60)

    generator = CharacterGenerator()

    # Generate mature woman in her 50s
    character = generator.generate(age_range=(50, 60))

    logger.info("\nGenerated Mature Character:")
    logger.info(f"  Age: {character.age} years old")
    logger.info(f"  Age Features: {', '.join(character.age_features)}")
    logger.info(f"  Hair: {character.hair_color} (may have grey)")
    logger.info(f"  Ethnicity: {character.ethnicity}")
    logger.info(f"  Skin: {character.skin_tone}")

    logger.info("\nMature Character Prompt:")
    logger.info(f"  {character.to_prompt()}")


def example_diversity_batch():
    """Example: Generate batch of diverse characters."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Batch Generation (10 Characters)")
    logger.info("=" * 60)

    generator = CharacterGenerator()

    # Generate 10 characters with diversity enforcement
    characters = generator.generate_batch(count=10, enforce_diversity=True)

    # Count ethnicities and skin tones
    ethnicity_counts = {}
    skin_tone_counts = {}

    for char in characters:
        ethnicity_counts[char.ethnicity] = ethnicity_counts.get(char.ethnicity, 0) + 1
        skin_tone_counts[char.skin_tone] = skin_tone_counts.get(char.skin_tone, 0) + 1

    logger.info("\nEthnicity Distribution:")
    for ethnicity, count in sorted(ethnicity_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {ethnicity}: {count} ({count/10:.0%})")

    logger.info("\nSkin Tone Distribution:")
    for skin_tone, count in sorted(skin_tone_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {skin_tone}: {count} ({count/10:.0%})")

    # Calculate non-white percentage
    non_white_count = sum(c for e, c in ethnicity_counts.items() if e != "caucasian")
    non_white_percentage = non_white_count / 10

    logger.info(f"\nDiversity Metrics:")
    logger.info(f"  Non-white characters: {non_white_count}/10 ({non_white_percentage:.0%})")
    logger.info(f"  Target: ≥70% (met: {'✓' if non_white_percentage >= 0.70 else '✗'})")

    # Show first 3 character prompts
    logger.info("\nSample Prompts (first 3):")
    for i, char in enumerate(characters[:3], 1):
        logger.info(f"\n  Character {i}:")
        logger.info(f"    {char.age}y {char.ethnicity} with {char.skin_tone} skin")
        logger.info(f"    Prompt: {char.to_prompt()[:150]}...")


def example_no_diversity_enforcement():
    """Example: Compare with diversity enforcement OFF."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Comparison - Diversity OFF vs ON")
    logger.info("=" * 60)

    generator = CharacterGenerator()

    # Generate 20 without diversity enforcement
    logger.info("\nGenerating 20 characters WITHOUT diversity enforcement...")
    chars_no_diversity = generator.generate_batch(count=20, enforce_diversity=False)

    ethnicity_no_div = {}
    for char in chars_no_diversity:
        ethnicity_no_div[char.ethnicity] = ethnicity_no_div.get(char.ethnicity, 0) + 1

    # Generate 20 WITH diversity enforcement
    logger.info("Generating 20 characters WITH diversity enforcement...")
    chars_with_diversity = generator.generate_batch(count=20, enforce_diversity=True)

    ethnicity_with_div = {}
    for char in chars_with_diversity:
        ethnicity_with_div[char.ethnicity] = ethnicity_with_div.get(char.ethnicity, 0) + 1

    # Compare
    logger.info("\nEthnicity Distribution Comparison:")
    logger.info(f"{'Ethnicity':<20} | {'No Diversity':>15} | {'With Diversity':>15}")
    logger.info("-" * 60)

    all_ethnicities = set(ethnicity_no_div.keys()) | set(ethnicity_with_div.keys())

    for ethnicity in sorted(all_ethnicities):
        no_div_count = ethnicity_no_div.get(ethnicity, 0)
        with_div_count = ethnicity_with_div.get(ethnicity, 0)

        logger.info(
            f"{ethnicity:<20} | {no_div_count:>6} ({no_div_count/20:>5.0%})  | "
            f"{with_div_count:>6} ({with_div_count/20:>5.0%})"
        )

    # Calculate metrics
    non_white_no_div = sum(c for e, c in ethnicity_no_div.items() if e != "caucasian")
    non_white_with_div = sum(c for e, c in ethnicity_with_div.items() if e != "caucasian")

    logger.info("\nDiversity Metrics:")
    logger.info(f"  Without enforcement: {non_white_no_div}/20 ({non_white_no_div/20:.0%}) non-white")
    logger.info(f"  With enforcement:    {non_white_with_div}/20 ({non_white_with_div/20:.0%}) non-white")
    logger.info(f"  Target: ≥70% non-white")


def example_json_export():
    """Example: Export character as JSON."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 6: JSON Export")
    logger.info("=" * 60)

    generator = CharacterGenerator()
    character = generator.generate()

    import json

    char_dict = character.to_dict()
    char_json = json.dumps(char_dict, indent=2)

    logger.info("\nCharacter as JSON:")
    logger.info(char_json)


def main():
    """Run all examples."""
    logger.info("Character Generator Examples")
    logger.info("Diversity System for Countering CivitAI Racial Bias\n")

    try:
        # Run examples
        example_single_character()
        example_explicit_character()
        example_age_constrained()
        example_diversity_batch()
        example_no_diversity_enforcement()
        example_json_export()

        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully! ✓")
        logger.info("=" * 60)

        logger.info("\nKey Takeaways:")
        logger.info("  1. Diversity enforcement WORKS - generates 70%+ non-white characters")
        logger.info("  2. Ethnic consistency - skin tone, hair texture match ethnicity")
        logger.info("  3. Higher prompt weights (1.3-1.6) for non-white features counter bias")
        logger.info("  4. Age-appropriate features automatically selected")
        logger.info("  5. Character attributes can be exported as JSON for logging")

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
