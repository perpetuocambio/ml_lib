"""
Example: ComfyUI Integration - Access 3,678+ LoRAs automatically.

This example shows how to use existing ComfyUI model library
without duplicating files.
"""

import sys
from pathlib import Path

# Add ml_lib to path if running standalone
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_lib.diffusion.config import (
    ComfyUIPathResolver,
    create_comfyui_registry,
    detect_comfyui_installation,
)
from ml_lib.diffusion.intelligent.hub_integration.entities import ModelType


def example_1_detect_and_stats():
    """Example 1: Auto-detect ComfyUI and get stats."""
    print("=" * 60)
    print("Example 1: Auto-detect ComfyUI Installation")
    print("=" * 60)

    # Auto-detect ComfyUI
    comfyui_path = detect_comfyui_installation()

    if comfyui_path:
        print(f"✅ ComfyUI found at: {comfyui_path}")

        # Create resolver
        resolver = ComfyUIPathResolver()

        # Get stats
        print("\n📊 Model Statistics:")
        stats = resolver.get_stats()
        for model_type, count in sorted(stats.items()):
            print(f"  {model_type:20s}: {count:6,d} models")

        # Resolve symlinks
        print("\n🔗 Resolved Paths:")
        resolved = resolver.resolve_symlinks()
        for model_type, path in sorted(resolved.items(), key=lambda x: x[0].value):
            print(f"  {model_type.value:20s}: {path}")

    else:
        print("❌ ComfyUI not found")
        print("   Set COMFYUI_PATH environment variable or install at /src/ComfyUI")


def example_2_scan_loras():
    """Example 2: Scan and list LoRAs."""
    print("\n" + "=" * 60)
    print("Example 2: Scan LoRA Library")
    print("=" * 60)

    resolver = ComfyUIPathResolver()
    loras = resolver.scan_models(ModelType.LORA)

    if loras:
        print(f"\n✅ Found {len(loras):,} LoRAs")
        print("\nFirst 10 LoRAs:")
        for i, lora in enumerate(loras[:10], 1):
            size_mb = lora.stat().st_size / (1024 * 1024)
            print(f"  {i:2d}. {lora.name:50s} ({size_mb:6.1f} MB)")

        # Check for metadata
        print("\n📝 Checking metadata:")
        loras_with_metadata = []
        for lora in loras[:20]:  # Check first 20
            metadata = resolver.get_comfyui_metadata(lora)
            if metadata:
                loras_with_metadata.append((lora.name, metadata))

        if loras_with_metadata:
            print(f"   Found {len(loras_with_metadata)} LoRAs with metadata")
            print(f"   Example: {loras_with_metadata[0][0]}")
            print(f"   Keys: {list(loras_with_metadata[0][1].keys())}")
        else:
            print("   No metadata files found")

    else:
        print("❌ No LoRAs found")


def example_3_create_registry():
    """Example 3: Create registry from ComfyUI."""
    print("\n" + "=" * 60)
    print("Example 3: Create Model Registry")
    print("=" * 60)

    print("\n⏳ Scanning and registering models (this may take a minute)...")

    # Create registry from ComfyUI
    # Only register critical model types to save time
    resolver = ComfyUIPathResolver()
    registry = resolver.create_registry_from_comfyui(
        model_types=[
            ModelType.CHECKPOINT,
            ModelType.LORA,
            ModelType.CONTROLNET,
            ModelType.VAE,
            ModelType.EMBEDDING,
        ]
    )

    print("\n✅ Registry created successfully!")

    # Get all LoRAs from registry
    loras = registry.get_all_loras()
    print(f"\n📚 Registered LoRAs: {len(loras):,}")

    if loras:
        print("\nSample LoRAs:")
        for i, lora in enumerate(loras[:5], 1):
            print(f"  {i}. {lora.model_id}")
            print(f"     Path: {lora.path}")
            if lora.metadata:
                print(f"     Metadata: {list(lora.metadata.keys())[:3]}...")

    # Get other model types
    checkpoints = registry.get_checkpoints()
    controlnets = registry.get_models_by_type(ModelType.CONTROLNET)
    vaes = registry.get_models_by_type(ModelType.VAE)

    print(f"\n📊 Summary:")
    print(f"   Checkpoints: {len(checkpoints)}")
    print(f"   LoRAs: {len(loras)}")
    print(f"   ControlNets: {len(controlnets)}")
    print(f"   VAEs: {len(vaes)}")


def example_4_intelligent_pipeline():
    """Example 4: Use with intelligent pipeline."""
    print("\n" + "=" * 60)
    print("Example 4: Intelligent Pipeline with ComfyUI Models")
    print("=" * 60)

    # Create registry from ComfyUI
    print("\n⏳ Creating registry from ComfyUI...")
    registry = create_comfyui_registry()

    # Create intelligent pipeline with ComfyUI registry
    print("⏳ Initializing intelligent pipeline...")
    try:
        from ml_lib.diffusion.intelligent.pipeline.services import (
            IntelligentGenerationPipeline,
        )
        from ml_lib.diffusion.intelligent.pipeline.entities import (
            PipelineConfig,
        )

        # Create config
        config = PipelineConfig(
            base_model="runwayml/stable-diffusion-v1-5",  # Example
        )

        # Create pipeline with ComfyUI registry
        pipeline = IntelligentGenerationPipeline(
            config=config,
            model_registry=registry,
        )

        print("\n✅ Pipeline ready with ComfyUI models!")

        # Example: Get recommendations
        print("\n💡 Getting recommendations for prompt...")
        recommendations = pipeline.analyze_and_recommend(
            "anime girl with cat ears and magical powers"
        )

        print(f"\n📋 Recommendations:")
        print(f"   Prompt complexity: {recommendations.prompt_analysis.complexity_score:.2f}")
        print(f"   Suggested LoRAs: {len(recommendations.suggested_loras)}")

        if recommendations.suggested_loras:
            print("\n🎯 Top LoRA recommendations:")
            for i, lora in enumerate(recommendations.suggested_loras[:3], 1):
                print(f"   {i}. {lora.lora_name}")
                print(f"      Confidence: {lora.confidence_score:.2f}")
                print(f"      Alpha: {lora.suggested_alpha:.2f}")
                if lora.reasoning:
                    print(f"      Reason: {lora.reasoning}")

        print(f"\n⚙️  Suggested parameters:")
        print(f"   Steps: {recommendations.suggested_params.num_steps}")
        print(f"   CFG: {recommendations.suggested_params.guidance_scale}")
        print(
            f"   Resolution: {recommendations.suggested_params.width}x{recommendations.suggested_params.height}"
        )

        print("\n🎉 Integration successful!")
        print("   Now you can use generate() to create images with optimal settings")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all dependencies are installed")


def example_5_quick_start():
    """Example 5: Quick start - one function call."""
    print("\n" + "=" * 60)
    print("Example 5: Quick Start (One-liner)")
    print("=" * 60)

    # One function to create registry
    registry = create_comfyui_registry()

    print(f"\n✅ Registry created with {len(registry.get_all_models())} models")
    print(f"   LoRAs: {len(registry.get_all_loras()):,}")
    print("\n💡 Usage:")
    print("   from ml_lib.diffusion.config import create_comfyui_registry")
    print("   registry = create_comfyui_registry()")
    print("   pipeline = IntelligentGenerationPipeline(model_registry=registry)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ComfyUI Integration Examples"
    )
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Which example to run (default: all)",
    )
    args = parser.parse_args()

    examples = {
        1: example_1_detect_and_stats,
        2: example_2_scan_loras,
        3: example_3_create_registry,
        4: example_4_intelligent_pipeline,
        5: example_5_quick_start,
    }

    if args.example:
        examples[args.example]()
    else:
        # Run all examples
        for example_func in examples.values():
            try:
                example_func()
            except Exception as e:
                print(f"\n❌ Example failed: {e}")
                import traceback

                traceback.print_exc()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
