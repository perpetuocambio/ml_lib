"""
Example usage of Efficient Memory Management (US 14.3).

This example demonstrates:
1. System resource detection
2. Memory usage tracking
3. Model pool with LRU eviction
4. Automatic offload strategies
"""

import logging
import time
from ml_lib.diffusion.intelligent.memory import (
    MemoryManager,
    ModelPool,
    ModelOffloader,
)
from ml_lib.diffusion.intelligent.memory.entities import (
    OffloadStrategy,
    EvictionPolicy,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def example_resource_detection():
    """Example: Detect system resources."""
    logger.info("=" * 60)
    logger.info("Example 1: Resource Detection")
    logger.info("=" * 60)

    manager = MemoryManager()
    resources = manager.resources

    logger.info("\nSystem Resources:")
    logger.info(f"  GPU Type: {resources.gpu_type}")
    logger.info(f"  Total VRAM: {resources.total_vram_gb:.2f} GB")
    logger.info(f"  Available VRAM: {resources.available_vram_gb:.2f} GB")
    logger.info(f"  Total RAM: {resources.total_ram_gb:.2f} GB")
    logger.info(f"  Available RAM: {resources.available_ram_gb:.2f} GB")
    logger.info(f"  VRAM Category: {resources.vram_category}")

    if resources.has_cuda:
        logger.info(f"  CUDA Devices: {resources.cuda_device_count}")
        logger.info(f"  Compute Capability: {resources.compute_capability}")


def example_memory_tracking():
    """Example: Track memory usage."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Memory Usage Tracking")
    logger.info("=" * 60)

    manager = MemoryManager()

    # Track memory usage
    logger.info("\nInitial Memory:")
    logger.info(f"  VRAM: {manager.get_vram_usage():.2f} GB")
    logger.info(f"  RAM: {manager.get_ram_usage():.2f} GB")

    # Simulate some work
    with manager.track_usage() as tracker:
        # Simulate memory allocation
        logger.info("\nDuring operation...")
        time.sleep(0.1)

    logger.info(f"\nPeak VRAM: {tracker.peak_vram:.2f} GB")

    # Get comprehensive report
    report = manager.get_memory_report()
    logger.info("\nMemory Report:")
    for key, value in report.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")


def example_model_pool():
    """Example: Model pool with LRU eviction."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Model Pool with LRU")
    logger.info("=" * 60)

    # Create pool with 10GB limit
    pool = ModelPool(max_size_gb=10.0, eviction_policy=EvictionPolicy.LRU)

    # Simulate loading models
    class DummyModel:
        def __init__(self, name):
            self.name = name

    def load_model(name):
        """Dummy model loader."""
        logger.info(f"  Loading {name}...")
        time.sleep(0.1)
        return DummyModel(name)

    # Load multiple models
    logger.info("\nLoading models into pool...")

    model1 = pool.load(
        "model_1",
        loader_fn=lambda: load_model("Model 1"),
        estimated_size_gb=3.0,
    )

    model2 = pool.load(
        "model_2",
        loader_fn=lambda: load_model("Model 2"),
        estimated_size_gb=4.0,
    )

    model3 = pool.load(
        "model_3",
        loader_fn=lambda: load_model("Model 3"),
        estimated_size_gb=5.0,  # Will trigger eviction
    )

    # Check stats
    stats = pool.get_stats()
    logger.info("\nPool Statistics:")
    logger.info(f"  Loaded models: {stats['loaded_count']}")
    logger.info(f"  Current size: {stats['current_size_gb']:.2f} GB")
    logger.info(f"  Max size: {stats['max_size_gb']:.2f} GB")
    logger.info(f"  Utilization: {stats['utilization']:.1%}")
    logger.info(f"  Models: {', '.join(stats['models'])}")


def example_offload_strategies():
    """Example: Different offload strategies."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Offload Strategies")
    logger.info("=" * 60)

    memory_manager = MemoryManager()

    strategies = [
        OffloadStrategy.AUTO,
        OffloadStrategy.CPU_OFFLOAD,
        OffloadStrategy.BALANCED,
        OffloadStrategy.SEQUENTIAL,
    ]

    for strategy in strategies:
        logger.info(f"\n{strategy.value.upper()} Strategy:")

        offloader = ModelOffloader(
            strategy=strategy,
            max_vram_gb=8.0,
            memory_manager=memory_manager,
        )

        config = offloader.get_offload_config()

        logger.info(f"  UNet: {config.unet_device}")
        logger.info(f"  Text Encoder: {config.text_encoder_device}")
        logger.info(f"  VAE: {config.vae_device}")
        logger.info(f"  LoRA: {config.lora_device}")

        if config.enable_sequential:
            logger.info("  Sequential loading: ENABLED")
        if config.enable_cpu_offload:
            logger.info("  CPU offload: ENABLED")

        logger.info(f"  Memory efficient: {config.memory_efficient}")


def example_low_vram_scenario():
    """Example: Optimal configuration for low VRAM."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Low VRAM Optimization")
    logger.info("=" * 60)

    manager = MemoryManager()
    vram = manager.resources.available_vram_gb

    logger.info(f"\nAvailable VRAM: {vram:.2f} GB")

    # Choose strategy based on VRAM
    if vram < 6:
        strategy = OffloadStrategy.SEQUENTIAL
        logger.info("Recommendation: SEQUENTIAL (extreme low VRAM)")
    elif vram < 8:
        strategy = OffloadStrategy.CPU_OFFLOAD
        logger.info("Recommendation: CPU_OFFLOAD (low VRAM)")
    elif vram < 12:
        strategy = OffloadStrategy.BALANCED
        logger.info("Recommendation: BALANCED (medium VRAM)")
    else:
        strategy = OffloadStrategy.FULL_GPU
        logger.info("Recommendation: FULL_GPU (high VRAM)")

    # Configure
    offloader = ModelOffloader(strategy=strategy, memory_manager=manager)
    config = offloader.get_offload_config()

    logger.info("\nOptimal Configuration:")
    logger.info(f"  Strategy: {strategy.value}")
    logger.info(f"  UNet device: {config.unet_device}")
    logger.info(f"  Text Encoder device: {config.text_encoder_device}")
    logger.info(f"  VAE device: {config.vae_device}")

    # Estimate what can fit
    estimated_base_model = 6.0  # GB for SDXL
    estimated_with_loras = estimated_base_model + 0.5  # Add 0.5GB for LoRAs

    can_fit_base = manager.can_fit_model(estimated_base_model)
    can_fit_with_loras = manager.can_fit_model(estimated_with_loras)

    logger.info("\nCapacity Estimates:")
    logger.info(f"  Can fit SDXL base model: {'✓' if can_fit_base else '✗'}")
    logger.info(f"  Can fit with LoRAs: {'✓' if can_fit_with_loras else '✗'}")


def example_complete_workflow():
    """Example: Complete memory management workflow."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 6: Complete Workflow")
    logger.info("=" * 60)

    # 1. Initialize managers
    memory_manager = MemoryManager()
    model_pool = ModelPool(max_size_gb=15.0)
    offloader = ModelOffloader(
        strategy=OffloadStrategy.AUTO, memory_manager=memory_manager
    )

    # 2. Get offload config
    config = offloader.get_offload_config()
    logger.info("\nOffload Configuration:")
    logger.info(f"  Strategy: AUTO -> {config}")

    # 3. Track memory during operation
    with memory_manager.track_usage() as tracker:
        logger.info("\nSimulating model operations...")

        # Simulate loading base model
        def load_base_model():
            time.sleep(0.1)
            return "BaseModelDummy"

        base_model = model_pool.load(
            "sdxl-base",
            loader_fn=load_base_model,
            estimated_size_gb=6.0,
        )

        # Simulate loading LoRA
        def load_lora():
            time.sleep(0.05)
            return "LoRADummy"

        lora = model_pool.load(
            "anime-lora", loader_fn=load_lora, estimated_size_gb=0.3
        )

    # 4. Report results
    logger.info("\nOperation Complete!")
    logger.info(f"  Peak VRAM: {tracker.peak_vram:.2f} GB")

    pool_stats = model_pool.get_stats()
    logger.info(f"  Models loaded: {pool_stats['loaded_count']}")
    logger.info(f"  Pool utilization: {pool_stats['utilization']:.1%}")

    # 5. Cleanup
    logger.info("\nCleaning up...")
    memory_manager.clear_cache()
    logger.info("  Cache cleared")


def main():
    """Run all examples."""
    logger.info("Efficient Memory Management Examples")
    logger.info("US 14.3: Resource Detection + Model Pool + Offloading\n")

    try:
        # Run examples
        example_resource_detection()
        example_memory_tracking()
        example_model_pool()
        example_offload_strategies()
        example_low_vram_scenario()
        example_complete_workflow()

        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully! ✓")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
