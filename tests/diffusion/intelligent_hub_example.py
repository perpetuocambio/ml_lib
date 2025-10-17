"""
Example usage of Model Hub Integration (US 14.1).

This example demonstrates:
1. Searching models on HuggingFace and CivitAI
2. Downloading and caching models
3. Managing models with ModelRegistry
"""

import logging
from ml_lib.diffusion.domain.services import (
    HuggingFaceHubService,
    CivitAIService,
    ModelRegistry,
)
from ml_lib.diffusion.models import (
    ModelFilter,
    ModelType,
    BaseModel,
    SortBy,
    Source,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def example_huggingface_search():
    """Example: Search models on HuggingFace."""
    logger.info("=" * 60)
    logger.info("Example 1: Searching HuggingFace Hub")
    logger.info("=" * 60)

    hf_service = HuggingFaceHubService()

    # Search for SDXL models
    filter = ModelFilter(task="text-to-image", library="diffusers")

    models = hf_service.search_models("stable-diffusion-xl", filter=filter, limit=5)

    logger.info(f"Found {len(models)} models:")
    for model in models:
        logger.info(f"  - {model.name} ({model.model_id})")
        logger.info(f"    Downloads: {model.download_count}")
        logger.info(f"    URL: {model.remote_url}")


def example_civitai_search():
    """Example: Search LoRAs on CivitAI."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Searching CivitAI for LoRAs")
    logger.info("=" * 60)

    civitai_service = CivitAIService()

    # Search for anime LoRAs for SDXL
    loras = civitai_service.search_models(
        query="anime",
        type=ModelType.LORA,
        base_model=BaseModel.SDXL,
        sort="Highest Rated",
        limit=5,
    )

    logger.info(f"Found {len(loras)} LoRAs:")
    for lora in loras:
        logger.info(f"  - {lora.name}")
        logger.info(f"    Rating: {lora.rating}/5.0")
        logger.info(f"    Downloads: {lora.download_count}")
        logger.info(f"    Trigger words: {', '.join(lora.trigger_words[:3])}")


def example_model_registry():
    """Example: Using ModelRegistry."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Using ModelRegistry")
    logger.info("=" * 60)

    registry = ModelRegistry()

    # Search across both sources
    logger.info("Searching for 'anime' across all sources...")
    results = registry.search(
        query="anime",
        sources=[Source.HUGGINGFACE, Source.CIVITAI],
        limit=10,
    )

    logger.info(f"Found {len(results)} models in registry")

    # List all LoRAs
    loras = registry.list_models(model_type=ModelType.LORA, limit=20)
    logger.info(f"Total LoRAs in registry: {len(loras)}")

    # List cached models
    cached = registry.hf_service.list_cached_models()
    logger.info(f"Cached HuggingFace models: {len(cached)}")


def example_download_model():
    """Example: Download a small test model."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Downloading a model")
    logger.info("=" * 60)

    hf_service = HuggingFaceHubService()

    # Download a tiny test model (for demonstration)
    logger.info("Downloading tiny-stable-diffusion for testing...")

    result = hf_service.download_model(
        "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
        allow_patterns=["*.json", "*.txt"],  # Only config files for this example
    )

    if result.success:
        logger.info(f"✓ Download successful!")
        logger.info(f"  Status: {result.status.value}")
        logger.info(f"  Path: {result.local_path}")
        logger.info(f"  Size: {result.download_mb:.2f} MB")
        if result.download_time_seconds > 0:
            logger.info(f"  Time: {result.download_time_seconds:.2f}s")
            logger.info(f"  Speed: {result.download_speed_mbps:.2f} MB/s")
    else:
        logger.error(f"✗ Download failed: {result.error_message}")


def example_unified_workflow():
    """Example: Complete workflow with registry."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Unified Workflow")
    logger.info("=" * 60)

    registry = ModelRegistry()

    # Step 1: Search for base model on HuggingFace
    logger.info("Step 1: Searching for SDXL base model...")
    hf_results = registry.hf_service.search_models("sdxl-base-1.0", limit=1)

    if hf_results:
        logger.info(f"  Found: {hf_results[0].name}")
        registry.register_model(hf_results[0])

    # Step 2: Search for complementary LoRAs on CivitAI
    logger.info("Step 2: Searching for anime LoRAs on CivitAI...")
    civitai_results = registry.civitai_service.search_models(
        query="anime style",
        type=ModelType.LORA,
        base_model=BaseModel.SDXL,
        limit=3,
    )

    for lora in civitai_results:
        logger.info(f"  Found LoRA: {lora.name}")
        registry.register_model(lora)

    # Step 3: Query registry for complete setup
    logger.info("Step 3: Querying registry for complete setup...")

    base_models = registry.list_models(
        model_type=ModelType.BASE_MODEL, base_model=BaseModel.SDXL, limit=5
    )
    loras = registry.list_models(
        model_type=ModelType.LORA, base_model=BaseModel.SDXL, limit=10
    )

    logger.info(f"  Available SDXL base models: {len(base_models)}")
    logger.info(f"  Available SDXL LoRAs: {len(loras)}")

    logger.info("\nReady for intelligent generation! 🎨")


def main():
    """Run all examples."""
    logger.info("Model Hub Integration Examples")
    logger.info("US 14.1: HuggingFace Hub + CivitAI Integration\n")

    try:
        # Run examples
        example_huggingface_search()
        example_civitai_search()
        example_model_registry()
        # example_download_model()  # Commented out - only run if you want to download
        example_unified_workflow()

        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully! ✓")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
