#!/usr/bin/env python3
"""
Simple test of the diffusion facade.

This test verifies that the facade can be imported and used with the
simplified API, without requiring a full generation pipeline.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_facade_import():
    """Test that the facade can be imported."""
    print("Testing facade import...")

    try:
        from ml_lib.diffusion import ImageGenerator, Generator, GenerationOptions
        print("✓ Successfully imported ImageGenerator, Generator, GenerationOptions")
        return True
    except ImportError as e:
        print(f"✗ Failed to import facade: {e}")
        return False


def test_generation_options():
    """Test that GenerationOptions can be created and configured."""
    print("\nTesting GenerationOptions...")

    try:
        from ml_lib.diffusion import GenerationOptions

        # Test default options
        default_opts = GenerationOptions()
        assert default_opts.steps == 35
        assert default_opts.cfg_scale == 7.5
        assert default_opts.width == 1024
        assert default_opts.height == 1024
        print("✓ Default options created successfully")

        # Test custom options
        custom_opts = GenerationOptions(
            steps=50,
            cfg_scale=8.0,
            width=512,
            height=512,
            seed=42,
            memory_mode="low",
            enable_learning=True
        )
        assert custom_opts.steps == 50
        assert custom_opts.cfg_scale == 8.0
        assert custom_opts.width == 512
        assert custom_opts.seed == 42
        assert custom_opts.memory_mode == "low"
        assert custom_opts.enable_learning is True
        print("✓ Custom options created successfully")

        return True
    except Exception as e:
        print(f"✗ Failed to create GenerationOptions: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generator_creation():
    """Test that ImageGenerator can be created (without initialization)."""
    print("\nTesting ImageGenerator creation...")

    try:
        from ml_lib.diffusion import ImageGenerator, GenerationOptions

        # Test default generator
        gen1 = ImageGenerator()
        assert gen1.model == "stabilityai/stable-diffusion-xl-base-1.0"
        assert gen1.device == "auto"
        assert gen1._pipeline is None  # Should be lazy-initialized
        print("✓ Default generator created successfully")

        # Test custom generator
        opts = GenerationOptions(steps=40, enable_loras=False)
        gen2 = ImageGenerator(
            model="runwayml/stable-diffusion-v1-5",
            device="cpu",
            options=opts
        )
        assert gen2.model == "runwayml/stable-diffusion-v1-5"
        assert gen2.device == "cpu"
        assert gen2.options.steps == 40
        assert gen2.options.enable_loras is False
        print("✓ Custom generator created successfully")

        # Test Generator alias
        from ml_lib.diffusion import Generator
        gen3 = Generator()
        assert isinstance(gen3, ImageGenerator)
        print("✓ Generator alias works correctly")

        return True
    except Exception as e:
        print(f"✗ Failed to create ImageGenerator: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_facade_api_signature():
    """Test that the facade has the expected API methods."""
    print("\nTesting facade API signature...")

    try:
        from ml_lib.diffusion import ImageGenerator

        gen = ImageGenerator()

        # Check that expected methods exist
        assert hasattr(gen, "generate_character")
        assert hasattr(gen, "generate_from_prompt")
        assert hasattr(gen, "analyze_prompt")
        assert hasattr(gen, "provide_feedback")
        print("✓ All expected methods are present")

        # Check method signatures (without calling them)
        import inspect

        # generate_character should accept optional parameters
        sig = inspect.signature(gen.generate_character)
        params = list(sig.parameters.keys())
        assert "age_range" in params
        assert "ethnicity" in params
        assert "style" in params
        assert "options" in params
        print("✓ generate_character has correct signature")

        # generate_from_prompt should accept prompt and options
        sig = inspect.signature(gen.generate_from_prompt)
        params = list(sig.parameters.keys())
        assert "prompt" in params
        assert "options" in params
        print("✓ generate_from_prompt has correct signature")

        return True
    except Exception as e:
        print(f"✗ Failed API signature test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_module_exports():
    """Test that the module exports the correct public API."""
    print("\nTesting module exports...")

    try:
        import ml_lib.diffusion as diff_module

        # Check __all__
        assert hasattr(diff_module, "__all__")
        expected_exports = ["ImageGenerator", "Generator", "GenerationOptions"]
        for export in expected_exports:
            assert export in diff_module.__all__, f"Missing export: {export}"
        print(f"✓ Module exports: {diff_module.__all__}")

        # Check version
        assert hasattr(diff_module, "__version__")
        print(f"✓ Module version: {diff_module.__version__}")

        return True
    except Exception as e:
        print(f"✗ Failed module exports test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all facade tests."""
    print("=" * 60)
    print("DIFFUSION FACADE SIMPLE TESTS")
    print("=" * 60)
    print()
    print("These tests verify the facade structure without requiring")
    print("a full diffusion pipeline or GPU.")
    print()

    tests = [
        ("Import", test_facade_import),
        ("GenerationOptions", test_generation_options),
        ("Generator Creation", test_generator_creation),
        ("API Signature", test_facade_api_signature),
        ("Module Exports", test_module_exports),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nThe facade is correctly structured and ready to use.")
        print("Next steps:")
        print("  1. Ensure diffusers/torch are installed for actual generation")
        print("  2. Run full integration tests with a model")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("The facade structure needs fixes before it can be used.")
        return 1


if __name__ == "__main__":
    exit(main())
