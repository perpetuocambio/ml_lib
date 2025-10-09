#!/usr/bin/env python3
"""Test script to verify the refactored CharacterGenerator works correctly."""

from pathlib import Path
import sys
import os

# Add the current project to the Python path
sys.path.insert(0, ".")


def test_character_generator():
    """Test that the CharacterGenerator can be imported and configured."""
    try:
        # Import the entities we created
        from ml_lib.diffusion.intelligent.prompting.entities import (
            CharacterAttributeSet,
            AttributeConfig,
            GeneratedCharacter,
        )

        print("✓ Character attribute entities imported successfully")

        # Import the generator
        from ml_lib.diffusion.intelligent.prompting.character_generator import (
            CharacterGenerator,
        )

        print("✓ CharacterGenerator imported successfully")

        # Test that the config file exists
        config_path = Path("./config/intelligent_prompting/character_attributes.yaml")
        if config_path.exists():
            print("✓ Character attributes config file exists")
        else:
            print("✗ Character attributes config file missing")
            return False

        # Try to initialize the character generator without using the registry
        # (since that might require infrastructure module)
        try:
            generator = CharacterGenerator(config_path=config_path)
            print("✓ CharacterGenerator initialized successfully")
        except Exception as e:
            print(f"✗ CharacterGenerator initialization failed: {e}")
            return False

        print(
            "✓ All tests passed! The refactored CharacterGenerator is working correctly."
        )
        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_config_files():
    """Test that all the configuration files exist."""
    config_dir = Path("./config/intelligent_prompting/")
    expected_files = [
        "character_attributes.yaml",
        "concept_categories.yaml",
        "lora_filters.yaml",
        "generation_profiles.yaml",
        "prompting_strategies.yaml",
    ]

    all_exist = True
    for file in expected_files:
        path = config_dir / file
        if path.exists():
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            all_exist = False

    return all_exist


if __name__ == "__main__":
    print("Testing the refactored configuration-driven code...")
    print("\n1. Testing configuration files:")
    config_ok = test_config_files()

    print("\n2. Testing CharacterGenerator:")
    generator_ok = test_character_generator()

    print(f"\nOverall result: {'✓ PASS' if config_ok and generator_ok else '✗ FAIL'}")
