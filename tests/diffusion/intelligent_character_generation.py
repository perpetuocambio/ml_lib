#!/usr/bin/env python3
"""Example usage of the intelligent character generator."""

from ml_lib.diffusion.intelligent.prompting import CharacterGenerator
from ml_lib.diffusion.intelligent.prompting.entities.generated_character import GeneratedCharacter


def main():
    """Demonstrate intelligent character generation."""

    # Create the generator
    generator = CharacterGenerator()

    # Example 1: Generate a mature woman with gothic style
    print("=== Example 1: Mature Gothic Woman ===")
    character1 = generator.generate_character()
    print(f"Generated character prompt: {character1.to_prompt()}")
    print()

    # Example 2: Generate a young adult with fantasy style
    print("=== Example 2: Young Fantasy Character ===")
    character2 = generator.generate_character()
    print(f"Generated character prompt: {character2.to_prompt()}")
    print()

    # Example 3: Generate a nurse character
    print("=== Example 3: Nurse Character ===")
    character3 = generator.generate_character()
    print(f"Generated character prompt: {character3.to_prompt()}")
    print()

    # Example 4: Generate multiple characters
    print("=== Example 4: Batch Generation ===")
    characters = generator.generate_batch(3)
    for i, character in enumerate(characters, 1):
        print(f"Character {i}: {character.to_prompt()[:100]}...")
    print()

    # Example 5: Demonstrate safety - blocked "schoolgirl" style
    print("=== Example 5: Safety Check ===")
    character5 = generator.generate_character()
    print(f"Character with blocked style replaced: {character5.to_prompt()}")
    print(
        "(Note: 'schoolgirl' style was automatically replaced with a safe alternative)"
    )


if __name__ == "__main__":
    main()
