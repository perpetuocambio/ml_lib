#!/usr/bin/env python3
"""Example usage of the intelligent character generator."""

from ml_lib.diffusion.intelligent.prompting.intelligent_generator import (
    IntelligentCharacterGenerator, CharacterGenerationContext
)


def main():
    """Demonstrate intelligent character generation."""
    
    # Create the generator
    generator = IntelligentCharacterGenerator()
    
    # Example 1: Generate a mature woman with gothic style
    print("=== Example 1: Mature Gothic Woman ===")
    context1 = CharacterGenerationContext(
        target_age=45,
        target_ethnicity="caucasian",
        target_style="goth",
        explicit_content_allowed=True,
        safety_level="strict"
    )
    
    character1 = generator.generate_character(context1)
    print(f"Generated character prompt: {character1.to_prompt()}")
    print()
    
    # Example 2: Generate a young adult with fantasy style
    print("=== Example 2: Young Fantasy Character ===")
    context2 = CharacterGenerationContext(
        target_age=25,
        target_ethnicity="east_asian",
        target_style="fantasy",
        explicit_content_allowed=True,
        safety_level="strict"
    )
    
    character2 = generator.generate_character(context2)
    print(f"Generated character prompt: {character2.to_prompt()}")
    print()
    
    # Example 3: Generate a nurse character
    print("=== Example 3: Nurse Character ===")
    context3 = CharacterGenerationContext(
        target_age=30,
        target_ethnicity="hispanic_latinx",
        target_style="nurse",
        explicit_content_allowed=True,
        safety_level="strict"
    )
    
    character3 = generator.generate_character(context3)
    print(f"Generated character prompt: {character3.to_prompt()}")
    print()
    
    # Example 4: Generate multiple characters
    print("=== Example 4: Batch Generation ===")
    context4 = CharacterGenerationContext(
        explicit_content_allowed=True,
        safety_level="strict"
    )
    
    characters = generator.generate_batch(3, context4)
    for i, character in enumerate(characters, 1):
        print(f"Character {i}: {character.to_prompt()[:100]}...")
    print()
    
    # Example 5: Demonstrate safety - blocked "schoolgirl" style
    print("=== Example 5: Safety Check ===")
    context5 = CharacterGenerationContext(
        target_style="schoolgirl",  # This should be blocked
        explicit_content_allowed=True,
        safety_level="strict"
    )
    
    # The system should automatically replace blocked styles with safe alternatives
    character5 = generator.generate_character(context5)
    print(f"Character with blocked style replaced: {character5.to_prompt()}")
    print("(Note: 'schoolgirl' style was automatically replaced with a safe alternative)")


if __name__ == "__main__":
    main()