"""
Demo of the new Clean Architecture.

This example shows how to use the refactored ml_lib with:
- Dependency Injection
- Rich domain entities
- Clean separation of layers
- Use cases instead of god classes
"""

from pathlib import Path

# Infrastructure setup
from ml_lib.infrastructure.di.container import DIContainer
from ml_lib.infrastructure.monitoring.resource_monitor_adapter import (
    ResourceMonitorAdapter,
)

# Domain
from ml_lib.diffusion.domain.entities.lora import LoRA, LoRARecommendation
from ml_lib.diffusion.domain.value_objects.weights import (
    LoRAWeight,
    ConfidenceScore,
    CFGScale,
)
from ml_lib.diffusion.domain.interfaces.resource_monitor import IResourceMonitor


def demo_value_objects():
    """Demo: Value Objects with validation."""
    print("=" * 60)
    print("DEMO 1: Value Objects")
    print("=" * 60)

    # Before: raw floats, no validation
    # alpha = -0.5  # BUG! Invalid value accepted

    # After: Value Objects guarantee validity
    try:
        weight = LoRAWeight(-0.5)  # Raises ValueError
    except ValueError as e:
        print(f"✅ Validation works: {e}")

    # Valid values
    weight = LoRAWeight(0.8)
    print(f"✅ Valid weight: {weight} (value={weight.value})")

    # Immutable
    try:
        weight.value = 1.5  # Raises exception
    except Exception:
        print("✅ Immutable: Cannot modify value")

    # Factory methods for convenience
    default = LoRAWeight.default()
    scaled = weight.scale_by(1.5)
    print(f"✅ Default: {default}, Scaled: {scaled}")

    # Other Value Objects
    confidence = ConfidenceScore.high()
    cfg = CFGScale.default_for_model("SDXL")
    print(f"✅ Confidence: {confidence}, CFG: {cfg}")
    print()


def demo_rich_entities(tmp_path: Path):
    """Demo: Rich entities with behavior."""
    print("=" * 60)
    print("DEMO 2: Rich Entities")
    print("=" * 60)

    # Create a fake LoRA file
    lora_file = tmp_path / "anime_style.safetensors"
    lora_file.write_text("fake lora")

    # Before: Anemic entity (just data)
    # lora_info = LoRAInfo(name="anime_style", alpha=0.8)
    # # No behavior, logic in services

    # After: Rich entity with behavior
    lora = LoRA.create(
        name="anime_style",
        path=lora_file,
        base_model="SDXL",
        weight=0.8,
        trigger_words=["anime", "manga"],
        tags=["illustration", "2d"],
        download_count=50000,
        rating=4.7,
    )

    # Entity knows how to validate itself
    print(f"✅ LoRA created: {lora}")

    # Entity has behavior
    prompt = "anime girl with magical powers"
    matches = lora.matches_prompt(prompt)
    relevance = lora.calculate_relevance(prompt)
    compatible = lora.is_compatible_with("SDXL")
    popularity = lora.get_popularity_score()

    print(f"✅ Matches prompt: {matches}")
    print(f"✅ Relevance score: {relevance}")
    print(f"✅ Compatible with SDXL: {compatible}")
    print(f"✅ Popularity: {popularity:.1f}/100")

    # Entity can transform itself
    scaled = lora.scale_weight(1.2)
    print(f"✅ Scaled weight: {scaled.weight}")
    print()


def demo_dependency_injection():
    """Demo: Dependency Injection."""
    print("=" * 60)
    print("DEMO 3: Dependency Injection")
    print("=" * 60)

    # Before: Hard-coded dependencies
    # class Service:
    #     def __init__(self):
    #         self.monitor = ResourceMonitor()  # Tight coupling

    # After: Dependency Injection
    container = DIContainer()
    container.register_singleton(IResourceMonitor, ResourceMonitorAdapter)

    # Container resolves dependencies automatically
    monitor = container.resolve(IResourceMonitor)
    stats = monitor.get_current_stats()

    print(f"✅ DI resolved IResourceMonitor")
    print(f"✅ Has GPU: {stats.has_gpu}")
    print(f"✅ Available VRAM: {stats.available_vram_gb:.1f}GB")

    # Easy to swap implementations (e.g., for testing)
    # container.register_singleton(IResourceMonitor, MockResourceMonitor)
    print()


def demo_domain_service(tmp_path: Path):
    """Demo: Domain service with rich entities."""
    print("=" * 60)
    print("DEMO 4: Domain Service")
    print("=" * 60)

    # Create sample LoRAs
    lora1_file = tmp_path / "anime.safetensors"
    lora1_file.write_text("fake")
    lora1 = LoRA.create(
        name="anime_style",
        path=lora1_file,
        base_model="SDXL",
        trigger_words=["anime"],
        tags=["illustration"],
    )

    lora2_file = tmp_path / "realistic.safetensors"
    lora2_file.write_text("fake")
    lora2 = LoRA.create(
        name="realistic",
        path=lora2_file,
        base_model="SDXL",
        trigger_words=["photorealistic"],
        tags=["photo"],
    )

    # Before: Service had all the logic
    # class LoRARecommender:
    #     def recommend(self, prompt):
    #         loras = self.registry.get_loras()
    #         scored = []
    #         for lora in loras:
    #             score = self._calculate_score(lora, prompt)  # Complex logic
    #             ...

    # After: Entity has the logic, service coordinates
    prompt = "anime girl illustration"

    # Entities calculate their own relevance
    rec1 = LoRARecommendation.create(lora=lora1, prompt=prompt)
    rec2 = LoRARecommendation.create(lora=lora2, prompt=prompt)

    print(f"✅ {rec1.lora.name}: {rec1.confidence} - {rec1.reasoning}")
    print(f"✅ {rec2.lora.name}: {rec2.confidence} - {rec2.reasoning}")

    # Service just filters and sorts
    recommendations = sorted(
        [rec1, rec2],
        key=lambda r: r.confidence.value,
        reverse=True,
    )
    print(f"✅ Best recommendation: {recommendations[0].lora.name}")
    print()


def demo_architecture_layers():
    """Demo: Clean architecture layers."""
    print("=" * 60)
    print("DEMO 5: Architecture Layers")
    print("=" * 60)

    print("Before: Everything mixed together")
    print("  ❌ IntelligentGenerationPipeline (774 lines)")
    print("     - Domain logic (LoRA matching)")
    print("     - Application logic (orchestration)")
    print("     - Infrastructure (DB access)")
    print()

    print("After: Clear separation")
    print("  ✅ Domain Layer:")
    print("     - LoRA entity (matching, scoring)")
    print("     - LoRARecommendationService (coordination)")
    print()
    print("  ✅ Application Layer:")
    print("     - GenerateImageUseCase (orchestrates)")
    print()
    print("  ✅ Infrastructure Layer:")
    print("     - ResourceMonitorAdapter (implements IResourceMonitor)")
    print("     - SQLiteLoRARepository (implements ILoRARepository)")
    print()

    print("Benefits:")
    print("  ✅ Testable (mock interfaces)")
    print("  ✅ Maintainable (small, focused classes)")
    print("  ✅ Flexible (swap implementations)")
    print()


if __name__ == "__main__":
    import tempfile

    print("\n")
    print("█" * 60)
    print("  NEW ARCHITECTURE DEMONSTRATION")
    print("█" * 60)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        demo_value_objects()
        demo_rich_entities(tmp_path)
        demo_dependency_injection()
        demo_domain_service(tmp_path)
        demo_architecture_layers()

    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print()
    print("Key Improvements:")
    print("  1. ✅ Value Objects prevent invalid states")
    print("  2. ✅ Rich entities encapsulate behavior")
    print("  3. ✅ DI enables testing and flexibility")
    print("  4. ✅ Domain services coordinate entities")
    print("  5. ✅ Clean layers separate concerns")
    print()
    print("Next: Migrate remaining services to this pattern")
    print()
