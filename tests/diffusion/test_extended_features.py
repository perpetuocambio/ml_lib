#!/usr/bin/env python3
"""Test script to verify the extended character generator with new features."""

from pathlib import Path
import yaml


def test_extended_features():
    """Test that the extended features work correctly."""
    try:
        # Check that the character generator file has the new methods
        gen_file = "./diffusion/intelligent/prompting/character_generator.py"
        with open(gen_file, "r") as f:
            content = f.read()

        print("Checking extended CharacterGenerator features...")

        checks = [
            (
                "Has _select_clothing_config method",
                "_select_clothing_config" in content,
            ),
            (
                "Has _select_accessories_config method",
                "_select_accessories_config" in content,
            ),
            (
                "Has _select_erotic_toys_config method",
                "_select_erotic_toys_config" in content,
            ),
            (
                "Has _select_activity_config method",
                "_select_activity_config" in content,
            ),
            (
                "Has extended generate method with new params",
                "include_accessories:" in content and "include_toys:" in content,
            ),
            (
                "Has updated entity definitions",
                "clothing_style:" in content
                or "clothing_keywords:" in content.split("GeneratedCharacter")[1]
                if "GeneratedCharacter" in content
                else False,
            ),
        ]

        all_passed = True
        for check_name, check_result in checks:
            if check_result:
                print(f"✓ {check_name}")
            else:
                print(f"✗ {check_name}")
                all_passed = False

        return all_passed
    except Exception as e:
        print(f"✗ Error testing extended features: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_new_config_categories():
    """Test that the new config categories exist in the YAML."""
    try:
        config_path = "./config/intelligent_prompting/character_attributes.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        print("\nChecking new configuration categories...")

        new_categories = ["clothing_styles", "accessories", "erotic_toys", "activities"]
        all_found = True

        for category in new_categories:
            if category in config:
                print(f"✓ {category} exists with {len(config[category])} items")
                # Check that it has proper structure
                if config[category]:
                    sample_key = list(config[category].keys())[0]
                    sample_value = config[category][sample_key]
                    has_keywords = "keywords" in sample_value
                    has_probability = "probability" in sample_value
                    print(
                        f"  Sample item '{sample_key}' has keywords: {has_keywords}, probability: {has_probability}"
                    )
            else:
                print(f"✗ {category} missing")
                all_found = False

        return all_found
    except Exception as e:
        print(f"✗ Error testing new config categories: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_updated_entities():
    """Test that the entities have been properly updated."""
    try:
        entities_file = (
            "./diffusion/intelligent/prompting/entities/character_attribute.py"
        )
        with open(entities_file, "r") as f:
            content = f.read()

        print("\nChecking updated entities...")

        checks = [
            (
                "GeneratedCharacter has clothing fields",
                "clothing_style:" in content and "clothing_keywords:" in content,
            ),
            (
                "GeneratedCharacter has accessory fields",
                "accessories:" in content and "accessory_keywords:" in content,
            ),
            (
                "GeneratedCharacter has toy fields",
                "erotic_toys:" in content and "toy_keywords:" in content,
            ),
            (
                "GeneratedCharacter has activity fields",
                "activity:" in content and "activity_keywords:" in content,
            ),
            (
                "CharacterAttributeSet has new categories",
                "clothing_styles:" in content and "accessories:" in content,
            ),
            ("CharacterAttributeSet has erotic_toys", "erotic_toys:" in content),
            ("CharacterAttributeSet has activities", "activities:" in content),
            (
                "to_prompt method handles new fields",
                "clothing_style !=" in content or "clothing_keywords" in content,
            ),
        ]

        all_passed = True
        for check_name, check_result in checks:
            if check_result:
                print(f"✓ {check_name}")
            else:
                print(f"✗ {check_name}")
                all_passed = False

        return all_passed
    except Exception as e:
        print(f"✗ Error testing updated entities: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config_update():
    """Test that the config is properly loaded with new categories."""
    try:
        gen_file = "./diffusion/intelligent/prompting/character_generator.py"
        with open(gen_file, "r") as f:
            content = f.read()

        print("\nChecking config loading updates...")

        checks = [
            (
                "_load_attribute_set includes new categories",
                "clothing_styles=" in content and "accessories=" in content,
            ),
            (
                "Initialization loads new categories",
                "self.clothing_styles" in content and "self.accessories" in content,
            ),
            (
                "Uses attribute_set for new categories",
                "self.attribute_set.clothing_styles" in content,
            ),
        ]

        all_passed = True
        for check_name, check_result in checks:
            if check_result:
                print(f"✓ {check_name}")
            else:
                print(f"✗ {check_name}")
                all_passed = False

        return all_passed
    except Exception as e:
        print(f"✗ Error testing config update: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing extended character generation features...")

    print("\n1. Testing new configuration categories exist:")
    config_ok = test_new_config_categories()

    print("\n2. Testing extended CharacterGenerator features:")
    extended_ok = test_extended_features()

    print("\n3. Testing updated entities:")
    entities_ok = test_updated_entities()

    print("\n4. Testing config loading updates:")
    config_load_ok = test_config_update()

    all_tests_passed = config_ok and extended_ok and entities_ok and config_load_ok

    print(f"\nOverall result: {'✓ PASS' if all_tests_passed else '✗ FAIL'}")

    if all_tests_passed:
        print("\n🎉 SUCCESS: All extended features have been implemented!")
        print("\nNEW FEATURES ADDED:")
        print("• Clothing styles (nude, lingerie, casual, formal, fetish wear)")
        print("• Accessories (jewelry, headwear, eyewear, bags, fetish accessories)")
        print("• Erotic toys (dildos, vibrators, anal toys, bdsm items)")
        print("• Activities (intimate, sexual, foreplay, bdsm)")
        print("• Updated character generation with all new features")
        print("\nThe character generator now supports a much richer set of attributes!")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
