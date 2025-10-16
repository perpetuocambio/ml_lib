"""Full Architecture Example - All Patterns Working Together.

This example demonstrates how all design patterns work together in harmony:
- Strategy Pattern (prompt analysis)
- Repository Pattern (data persistence)
- Command Pattern (write operations)
- Query Pattern (read operations)
- Observer Pattern (event notifications)

Complete workflow:
1. User requests image generation (Command)
2. Prompt is analyzed (Strategy Pattern)
3. LoRAs are queried (Query Pattern)
4. Image is generated (Use Case)
5. Events are published (Observer Pattern)
6. Handlers react (logging, metrics, notifications)
"""

import asyncio
from ml_lib.diffusion.domain.value_objects import (
    PromptText,
    BaseModel,
)
from ml_lib.diffusion.domain.entities.lora import LoRA, LoRATriggerWord
from ml_lib.diffusion.infrastructure.repositories.in_memory_lora_repository import (
    InMemoryLoRARepository,
)
from ml_lib.diffusion.domain.services.lora_recommendation_service import (
    LoRARecommendationService,
)

# Commands & Queries
from ml_lib.diffusion.application.commands import (
    CommandBus,
    RecommendLoRAsCommand,
    RecommendLoRAsHandler,
)
from ml_lib.diffusion.application.queries import (
    QueryBus,
    GetAllLoRAsQuery,
    GetLoRAsByBaseModelQuery,
    GetAllLoRAsHandler,
    GetLoRAsByBaseModelHandler,
)

# Events
from ml_lib.diffusion.domain.events import (
    EventBus,
    LoRAsRecommendedEvent,
    LoRALoadedEvent,
    LoggingEventHandler,
    MetricsEventHandler,
    CachingHandler,
)


async def main():
    """
    Demonstrate full architecture with all patterns integrated.
    """
    print("=" * 70)
    print("🏗️  FULL ARCHITECTURE EXAMPLE - ALL PATTERNS WORKING TOGETHER")
    print("=" * 70)
    print()

    # ========================================
    # 1. SETUP: Initialize Infrastructure
    # ========================================
    print("📦 Step 1: Setting up infrastructure...")
    print()

    # Repository Pattern: Data persistence
    repository = InMemoryLoRARepository()

    # Add sample LoRAs
    sample_loras = [
        LoRA(
            id="anime-style-v1",
            name="Anime Art Style V1",
            base_model=BaseModel("SDXL"),
            trigger_words=[
                LoRATriggerWord("anime"),
                LoRATriggerWord("manga style"),
            ],
            strength=0.8,
            filename="anime-style-v1.safetensors",
        ),
        LoRA(
            id="detailed-portrait-v2",
            name="Detailed Portrait V2",
            base_model=BaseModel("SDXL"),
            trigger_words=[
                LoRATriggerWord("portrait"),
                LoRATriggerWord("detailed face"),
            ],
            strength=0.7,
            filename="portrait-v2.safetensors",
        ),
        LoRA(
            id="cyberpunk-city-v1",
            name="Cyberpunk City V1",
            base_model=BaseModel("SDXL"),
            trigger_words=[
                LoRATriggerWord("cyberpunk"),
                LoRATriggerWord("neon city"),
            ],
            strength=0.9,
            filename="cyberpunk-v1.safetensors",
        ),
    ]

    for lora in sample_loras:
        repository.add_lora(lora)

    print(f"✅ Repository initialized with {len(sample_loras)} LoRAs")
    print()

    # Domain Service
    lora_service = LoRARecommendationService(repository)

    # ========================================
    # 2. OBSERVER PATTERN: Setup Event Bus
    # ========================================
    print("📡 Step 2: Setting up Observer Pattern (Event Bus)...")
    print()

    event_bus = EventBus()

    # Subscribe handlers to events
    logging_handler = LoggingEventHandler()
    metrics_handler = MetricsEventHandler()
    caching_handler = CachingHandler()

    event_bus.subscribe(LoRAsRecommendedEvent, logging_handler)
    event_bus.subscribe(LoRALoadedEvent, caching_handler)

    print("✅ Event Bus configured with 2 handlers")
    print("   - LoggingEventHandler → LoRAsRecommendedEvent")
    print("   - CachingHandler → LoRALoadedEvent")
    print()

    # ========================================
    # 3. CQRS: Setup Command and Query Buses
    # ========================================
    print("🚌 Step 3: Setting up CQRS (Command & Query Buses)...")
    print()

    # Command Bus (write operations)
    command_bus = CommandBus()
    command_bus.register(
        RecommendLoRAsCommand,
        RecommendLoRAsHandler(lora_service, event_bus)
    )

    # Query Bus (read operations)
    query_bus = QueryBus()
    query_bus.register(GetAllLoRAsQuery, GetAllLoRAsHandler(lora_service))
    query_bus.register(
        GetLoRAsByBaseModelQuery,
        GetLoRAsByBaseModelHandler(lora_service)
    )

    print("✅ Command Bus registered: RecommendLoRAsCommand")
    print("✅ Query Bus registered: GetAllLoRAsQuery, GetLoRAsByBaseModelQuery")
    print()

    # ========================================
    # 4. QUERY PATTERN: Read Operations
    # ========================================
    print("🔍 Step 4: Executing Queries (Read Operations)...")
    print()

    # Query 1: Get all LoRAs
    print("Query: Get all LoRAs")
    all_loras_query = GetAllLoRAsQuery()
    all_loras_result = query_bus.dispatch(all_loras_query)
    print(f"✅ Found {len(all_loras_result.data)} LoRAs")
    for lora in all_loras_result.data:
        print(f"   - {lora.name} ({lora.id})")
    print()

    # Query 2: Get LoRAs by base model
    print("Query: Get LoRAs for SDXL")
    sdxl_query = GetLoRAsByBaseModelQuery(base_model="SDXL")
    sdxl_result = query_bus.dispatch(sdxl_query)
    print(f"✅ Found {len(sdxl_result.data)} SDXL LoRAs")
    print()

    # ========================================
    # 5. COMMAND PATTERN: Write Operations
    # ========================================
    print("✍️  Step 5: Executing Commands (Write Operations)...")
    print()

    # Command: Recommend LoRAs
    print("Command: Recommend LoRAs for 'anime girl portrait'")
    recommend_command = RecommendLoRAsCommand(
        prompt="anime girl portrait",
        base_model="SDXL",
        max_loras=3,
    )
    recommend_result = command_bus.dispatch(recommend_command)

    if recommend_result.is_success:
        recommendations = recommend_result.data
        print(f"✅ Command succeeded: {len(recommendations)} recommendations")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec.lora.name} (confidence: {rec.confidence:.2f})")
    else:
        print(f"❌ Command failed: {recommend_result.error}")
    print()

    # ========================================
    # 6. OBSERVER PATTERN: Events Published
    # ========================================
    print("📢 Step 6: Events were published during command execution!")
    print()
    print("Events that were published:")
    print("   - LoRAsRecommendedEvent → LoggingEventHandler logged it")
    print()

    # Manually publish another event to demonstrate
    print("Publishing manual event: LoRALoadedEvent")
    load_event = LoRALoadedEvent.create(
        lora_id="anime-style-v1",
        lora_name="Anime Art Style V1",
        base_model="SDXL",
    )
    await event_bus.publish(load_event)
    print("✅ Event published and handled")
    print()

    # Show caching handler statistics
    most_loaded = caching_handler.get_most_loaded_loras(top_n=3)
    if most_loaded:
        print("📊 Most loaded LoRAs (from CachingHandler):")
        for lora_id, count in most_loaded:
            print(f"   - {lora_id}: {count} loads")
    print()

    # ========================================
    # 7. SUMMARY
    # ========================================
    print("=" * 70)
    print("🎊 SUMMARY - All Patterns Working Together!")
    print("=" * 70)
    print()
    print("✅ Repository Pattern: Data persistence with InMemoryLoRARepository")
    print("✅ Strategy Pattern: Used internally by PromptAnalyzer")
    print("✅ Command Pattern: RecommendLoRAsCommand executed via CommandBus")
    print("✅ Query Pattern: GetAllLoRAsQuery & GetLoRAsByBaseModelQuery")
    print("✅ Observer Pattern: Events published to EventBus, handlers reacted")
    print()
    print("🏗️  Architecture layers:")
    print("   Application → Commands/Queries → Domain Services → Repository")
    print("   Events → EventBus → Handlers (cross-cutting concerns)")
    print()
    print("💡 Benefits:")
    print("   - Loose coupling between components")
    print("   - Clear separation of concerns (CQRS)")
    print("   - Event-driven notifications")
    print("   - Testable and maintainable code")
    print("   - SOLID principles applied throughout")
    print()
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
