"""
Example usage of Intelligent Prompting System (US 14.2).

This example demonstrates:
1. Semantic prompt analysis with Ollama
2. Intelligent LoRA recommendation
3. Automatic parameter optimization
"""

import logging
from ml_lib.diffusion.services import (
    PromptAnalyzer,
    LoRARecommender,
    ParameterOptimizer,
    ModelRegistry,
)
from ml_lib.diffusion.generation.parameter_optimizer import (
    GenerationConstraints,
)
from ml_lib.diffusion.models import Priority

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def example_prompt_analysis():
    """Example: Analyze a prompt with Ollama."""
    logger.info("=" * 60)
    logger.info("Example 1: Prompt Analysis")
    logger.info("=" * 60)

    # Initialize analyzer (will use Ollama if available)
    analyzer = PromptAnalyzer(
        ollama_url="http://localhost:11434",
        model_name="llama2",
        use_llm=False,  # Set to False for rule-based only
    )

    # Analyze a complex prompt
    prompt = (
        "anime girl with magical powers, detailed Victorian mansion background, "
        "cinematic lighting, masterpiece, best quality, intricate details"
    )

    analysis = analyzer.analyze(prompt)

    # Display results
    logger.info(f"Prompt: {prompt}\n")
    logger.info("Analysis Results:")
    logger.info(
        f"  Complexity: {analysis.complexity_category.value} ({analysis.complexity_score:.2f})"
    )
    logger.info(f"  Total concepts detected: {analysis.concept_count}")

    logger.info("\n  Detected Concepts by Category:")
    for category, concepts in analysis.detected_concepts.items():
        logger.info(f"    {category}: {', '.join(concepts)}")

    if analysis.intent:
        logger.info("\n  Detected Intent:")
        logger.info(f"    Style: {analysis.intent.artistic_style.value}")
        logger.info(f"    Content: {analysis.intent.content_type.value}")
        logger.info(f"    Quality: {analysis.intent.quality_level.value}")
        logger.info(f"    Confidence: {analysis.intent.confidence:.2f}")


def example_lora_recommendation():
    """Example: Get LoRA recommendations."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: LoRA Recommendation")
    logger.info("=" * 60)

    # Initialize components
    registry = ModelRegistry()
    analyzer = PromptAnalyzer(use_llm=False)  # Use rule-based for speed
    recommender = LoRARecommender(registry=registry)

    # Analyze prompt
    prompt = "anime style magical girl with glowing effects, fantasy background"
    analysis = analyzer.analyze(prompt)

    logger.info(f"Prompt: {prompt}\n")

    # Get recommendations
    recommendations = recommender.recommend(
        prompt_analysis=analysis,
        base_model="sdxl-base-1.0",
        max_loras=3,
        min_confidence=0.4,
    )

    logger.info(f"Found {len(recommendations)} LoRA recommendations:\n")

    for i, rec in enumerate(recommendations, 1):
        logger.info(f"{i}. {rec.lora_name}")
        logger.info(f"   Confidence: {rec.confidence_score:.2f}")
        logger.info(f"   Suggested weight (alpha): {rec.suggested_alpha:.2f}")
        logger.info(f"   Matching concepts: {', '.join(rec.matching_concepts[:3])}")
        logger.info(f"   Reasoning: {rec.reasoning}\n")


def example_parameter_optimization():
    """Example: Optimize generation parameters."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Parameter Optimization")
    logger.info("=" * 60)

    analyzer = PromptAnalyzer(use_llm=False)
    optimizer = ParameterOptimizer()

    # Test different prompts and priorities
    test_cases = [
        ("cat", Priority.SPEED, "Simple prompt, speed priority"),
        (
            "anime girl, blue eyes, detailed outfit",
            Priority.BALANCED,
            "Moderate complexity, balanced",
        ),
        (
            "photorealistic portrait of Victorian noble, intricate lace, "
            "Rembrandt lighting, 85mm lens, shallow DOF, masterpiece",
            Priority.QUALITY,
            "Complex prompt, quality priority",
        ),
    ]

    for prompt, priority, description in test_cases:
        logger.info(f"\n{description}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Priority: {priority.value}\n")

        # Analyze
        analysis = analyzer.analyze(prompt)

        # Optimize
        constraints = GenerationConstraints(
            max_time_seconds=120,
            max_vram_gb=12,
            priority=priority,
        )

        params = optimizer.optimize(analysis, constraints)

        # Display
        logger.info("Optimized Parameters:")
        logger.info(f"  Steps: {params.num_steps}")
        logger.info(f"  CFG Scale: {params.guidance_scale}")
        logger.info(f"  Resolution: {params.width}x{params.height}")
        logger.info(f"  Sampler: {params.sampler_name}")
        logger.info(f"  Clip Skip: {params.clip_skip}")

        logger.info("\nEstimates:")
        logger.info(f"  Time: {params.estimated_time_seconds:.1f}s")
        logger.info(f"  VRAM: {params.estimated_vram_gb:.2f} GB")
        logger.info(f"  Quality Score: {params.estimated_quality_score:.2f}")


def example_complete_workflow():
    """Example: Complete workflow from prompt to optimized config."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Complete Workflow")
    logger.info("=" * 60)

    # Initialize all components
    registry = ModelRegistry()
    analyzer = PromptAnalyzer(use_llm=False)
    recommender = LoRARecommender(registry=registry)
    optimizer = ParameterOptimizer()

    # User prompt
    prompt = (
        "anime girl with magical powers, elegant pose, "
        "detailed Victorian mansion interior, cinematic lighting, masterpiece"
    )

    logger.info(f"User Prompt: {prompt}\n")

    # Step 1: Analyze prompt
    logger.info("Step 1: Analyzing prompt...")
    analysis = analyzer.analyze(prompt)
    logger.info(f"  Complexity: {analysis.complexity_category.value}")
    logger.info(
        f"  Style: {analysis.intent.artistic_style.value if analysis.intent else 'unknown'}"
    )

    # Step 2: Recommend LoRAs
    logger.info("\nStep 2: Recommending LoRAs...")
    loras = recommender.recommend(analysis, base_model="sdxl-base-1.0", max_loras=3)
    logger.info(f"  Selected {len(loras)} LoRAs:")
    for lora in loras:
        logger.info(f"    - {lora.lora_name} (α={lora.suggested_alpha:.2f})")

    # Step 3: Optimize parameters
    logger.info("\nStep 3: Optimizing parameters...")
    constraints = GenerationConstraints(priority=Priority.QUALITY)
    params = optimizer.optimize(analysis, constraints)
    logger.info(f"  Steps: {params.num_steps}")
    logger.info(f"  CFG: {params.guidance_scale}")
    logger.info(f"  Resolution: {params.width}x{params.height}")
    logger.info(f"  Sampler: {params.sampler_name}")

    # Step 4: Summary
    logger.info("\n" + "=" * 40)
    logger.info("Generation Configuration Ready!")
    logger.info("=" * 40)
    logger.info(f"Estimated time: {params.estimated_time_seconds:.1f}s")
    logger.info(f"Estimated VRAM: {params.estimated_vram_gb:.2f} GB")
    logger.info(f"Quality score: {params.estimated_quality_score:.2f}")


def main():
    """Run all examples."""
    logger.info("Intelligent Prompting System Examples")
    logger.info(
        "US 14.2: Semantic Analysis + LoRA Recommendation + Parameter Optimization\n"
    )

    try:
        # Run examples
        example_prompt_analysis()
        example_lora_recommendation()
        example_parameter_optimization()
        example_complete_workflow()

        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully! ✓")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
