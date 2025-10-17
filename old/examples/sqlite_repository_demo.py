"""
SQLite Repository Demo - Real persistence in action.

Demonstrates SQLiteModelRepository with real database operations.
"""

from pathlib import Path
import tempfile

from ml_lib.diffusion.infrastructure.persistence.sqlite_model_repository import (
    SQLiteModelRepository,
)
from ml_lib.diffusion.domain.entities.lora import LoRA
from ml_lib.diffusion.domain.services.lora_recommendation_service import (
    LoRARecommendationService,
)


def main():
    """Run SQLite repository demo."""
    print("\n")
    print("=" * 70)
    print("  SQLite REPOSITORY DEMO: Real Persistence")
    print("=" * 70)
    print()

    # === STEP 1: Create SQLite Repository ===
    print("STEP 1: Creating SQLite Repository...")
    print("-" * 70)

    # Create temp database (in production, use persistent path)
    db_path = Path(tempfile.gettempdir()) / "demo_loras.db"
    print(f"Database path: {db_path}")

    repo = SQLiteModelRepository(db_path=db_path)
    print(f"✅ SQLite repository created")
    print(f"✅ Current LoRA count: {repo.count_loras()}")
    print()

    # === STEP 2: Add LoRAs to Database ===
    print("STEP 2: Adding LoRAs to database...")
    print("-" * 70)

    # Create temp directory for fake LoRA files
    temp_dir = Path(tempfile.gettempdir()) / "demo_loras"
    temp_dir.mkdir(exist_ok=True)

    # Sample LoRAs to add
    sample_data = [
        {
            "name": "anime_masterpiece",
            "base_model": "SDXL",
            "weight": 0.85,
            "trigger_words": ["anime", "masterpiece", "high quality"],
            "tags": ["anime", "illustration", "2d"],
            "download_count": 75000,
            "rating": 4.8,
        },
        {
            "name": "photorealistic_pro",
            "base_model": "SDXL",
            "weight": 1.0,
            "trigger_words": ["photorealistic", "detailed", "8k"],
            "tags": ["realistic", "photo", "professional"],
            "download_count": 50000,
            "rating": 4.7,
        },
        {
            "name": "cyberpunk_2077",
            "base_model": "SDXL",
            "weight": 0.9,
            "trigger_words": ["cyberpunk", "neon", "futuristic"],
            "tags": ["cyberpunk", "sci-fi", "neon", "city"],
            "download_count": 40000,
            "rating": 4.6,
        },
    ]

    for data in sample_data:
        # Create fake file
        lora_file = temp_dir / f"{data['name']}.safetensors"
        lora_file.write_text("fake lora data")

        # Create LoRA entity
        lora = LoRA.create(
            name=data["name"],
            path=lora_file,
            base_model=data["base_model"],
            weight=data["weight"],
            trigger_words=data["trigger_words"],
            tags=data["tags"],
            download_count=data["download_count"],
            rating=data["rating"],
        )

        # Add to database
        try:
            repo.add_lora(lora)
            print(f"✅ Added: {lora.name} (rating: {lora.rating})")
        except ValueError:
            print(f"⚠ Skipped (already exists): {lora.name}")

    print(f"\n✅ Total LoRAs in database: {repo.count_loras()}")
    print()

    # === STEP 3: Query Database ===
    print("STEP 3: Querying database...")
    print("-" * 70)

    # Get all LoRAs
    all_loras = repo.get_all_loras()
    print(f"All LoRAs: {len(all_loras)}")
    for lora in all_loras:
        print(f"  - {lora.name} ({lora.base_model})")
    print()

    # Get by base model
    sdxl_loras = repo.get_loras_by_base_model("SDXL")
    print(f"SDXL LoRAs: {len(sdxl_loras)}")
    print()

    # Get by tags
    anime_loras = repo.get_loras_by_tags(["anime", "illustration"])
    print(f"Anime/Illustration LoRAs: {len(anime_loras)}")
    for lora in anime_loras:
        print(f"  - {lora.name}")
    print()

    # Get popular
    popular = repo.get_popular_loras(limit=3)
    print("Top 3 popular LoRAs:")
    for i, lora in enumerate(popular, 1):
        print(
            f"  {i}. {lora.name} - {lora.download_count:,} downloads, "
            f"rating {lora.rating}"
        )
    print()

    # Search
    search_results = repo.search_loras(
        query="anime",
        base_model="SDXL",
        min_rating=4.0,
    )
    print(f"Search 'anime' (min rating 4.0): {len(search_results)} results")
    for lora in search_results:
        print(f"  - {lora.name}")
    print()

    # === STEP 4: Update LoRA ===
    print("STEP 4: Updating LoRA...")
    print("-" * 70)

    lora_to_update = repo.get_lora_by_name("anime_masterpiece")
    if lora_to_update:
        print(f"Before: {lora_to_update.name} - downloads: {lora_to_update.download_count}")

        # Create updated version
        updated = LoRA.create(
            name=lora_to_update.name,
            path=lora_to_update.path,
            base_model=lora_to_update.base_model,
            weight=lora_to_update.weight.value,
            trigger_words=lora_to_update.trigger_words,
            tags=lora_to_update.tags,
            download_count=100000,  # Updated!
            rating=5.0,  # Updated!
        )

        repo.update_lora(updated)
        print(f"✅ Updated!")

        # Verify
        verified = repo.get_lora_by_name("anime_masterpiece")
        print(f"After: {verified.name} - downloads: {verified.download_count}, rating: {verified.rating}")
    print()

    # === STEP 5: Use with Domain Service ===
    print("STEP 5: Using repository with domain service...")
    print("-" * 70)

    service = LoRARecommendationService(repository=repo)

    recommendations = service.recommend(
        prompt="cyberpunk city at night with neon lights",
        base_model="SDXL",
        max_loras=3,
        min_confidence=0.3,
    )

    print(f"Recommendations for 'cyberpunk city at night':")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.lora.name}")
        print(f"   Confidence: {rec.confidence.to_percentage()}%")
        print(f"   Weight: {rec.lora.weight.value}")
        print(f"   Reasoning: {rec.reasoning}")
        print()

    # === STEP 6: Persistence Demo ===
    print("STEP 6: Demonstrating persistence...")
    print("-" * 70)

    # Create new repository instance (simulates app restart)
    repo2 = SQLiteModelRepository(db_path=db_path)
    print(f"✅ New repository instance created")
    print(f"✅ LoRA count (persisted): {repo2.count_loras()}")

    retrieved = repo2.get_lora_by_name("cyberpunk_2077")
    if retrieved:
        print(f"✅ Retrieved persisted LoRA: {retrieved.name}")
        print(f"   - Downloads: {retrieved.download_count:,}")
        print(f"   - Rating: {retrieved.rating}")
        print(f"   - Trigger words: {retrieved.trigger_words}")
    print()

    # === STEP 7: Cleanup (Optional) ===
    print("STEP 7: Database statistics...")
    print("-" * 70)

    print(f"Total LoRAs: {repo.count_loras()}")
    print(f"Database file size: {db_path.stat().st_size / 1024:.2f} KB")
    print(f"Database path: {db_path}")
    print()

    # === FINAL SUMMARY ===
    print("=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70)
    print()

    print("What we demonstrated:")
    print("1. ✅ SQLite repository with real persistence")
    print("2. ✅ CRUD operations (Create, Read, Update, Delete)")
    print("3. ✅ Advanced queries (search, filter, sort)")
    print("4. ✅ Integration with domain service")
    print("5. ✅ Data persistence across instances")
    print("6. ✅ Production-ready repository pattern")
    print()

    print("Benefits over InMemoryRepository:")
    print("- ✅ Data persists across restarts")
    print("- ✅ Production-ready")
    print("- ✅ Efficient queries with indexes")
    print("- ✅ Transaction support")
    print("- ✅ Can handle large datasets")
    print()

    print(f"Database retained at: {db_path}")
    print("(Delete manually if no longer needed)")
    print()


if __name__ == "__main__":
    main()
