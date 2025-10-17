"""Test ProcessedPrompt value object."""

import logging

logging.basicConfig(level=logging.INFO)

print("=" * 80)
print("TEST: ProcessedPrompt Value Object")
print("=" * 80)

# Test 1: Import
try:
    # Import directly from modules to avoid circular imports
    from ml_lib.diffusion.domain.value_objects_models.value_objects.processed_prompt import ProcessedPrompt
    from ml_lib.diffusion.domain.value_objects_models.content_tags import (
        TokenClassification,
        PromptTokenPriority,
        NSFWCategory,
    )
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Create ProcessedPrompt without changes
print("\n" + "=" * 80)
print("TEST 2: ProcessedPrompt without modifications")
print("=" * 80)

try:
    processed = ProcessedPrompt(
        original="1girl, beautiful, masterpiece",
        final="1girl, beautiful, masterpiece",
        original_token_count=10,
        final_token_count=10,
        was_modified=False,
        architecture="SDXL",
        quality_level="high",
    )

    print(f"Original: {processed.original}")
    print(f"Final: {processed.final}")
    print(f"Was modified: {processed.was_modified}")
    print(f"Message: {processed.get_user_message()}")
    print(f"Short summary: {processed.get_short_summary()}")

    assert not processed.has_critical_loss
    assert not processed.has_high_priority_loss
    assert processed.token_reduction_ratio == 1.0
    assert processed.tokens_removed == 0

    print("✅ No-modification case works correctly")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Create ProcessedPrompt with modifications
print("\n" + "=" * 80)
print("TEST 3: ProcessedPrompt with compaction")
print("=" * 80)

try:
    # Simulate removed tokens
    removed_tokens = [
        TokenClassification(
            token="straight hair",
            priority=PromptTokenPriority.MEDIUM,
            category=None,
            is_quality_tag=False,
        ),
        TokenClassification(
            token="long hair",
            priority=PromptTokenPriority.MEDIUM,
            category=None,
            is_quality_tag=False,
        ),
        TokenClassification(
            token="masterpiece",
            priority=PromptTokenPriority.LOW,
            category=None,
            is_quality_tag=True,
        ),
    ]

    processed = ProcessedPrompt(
        original="1girl, brown hair, straight hair, long hair, perfect body, fellatio, masterpiece, best quality",
        final="1girl, brown hair, perfect body, fellatio, best quality",
        original_token_count=20,
        final_token_count=12,
        was_modified=True,
        removed_tokens=removed_tokens,
        architecture="SDXL",
        quality_level="balanced",
    )

    processed.add_modification("Added quality tags for SDXL")
    processed.add_modification("Compacted from 20 to 12 tokens")
    processed.add_warning("Some descriptive tokens were removed")

    print(f"Original tokens: {processed.original_token_count}")
    print(f"Final tokens: {processed.final_token_count}")
    print(f"Reduction ratio: {processed.token_reduction_ratio:.1%}")
    print(f"Tokens removed: {processed.tokens_removed}")
    print(f"Has critical loss: {processed.has_critical_loss}")
    print(f"Has high priority loss: {processed.has_high_priority_loss}")

    print("\nUser message:")
    print(processed.get_user_message())

    print(f"\nShort summary: {processed.get_short_summary()}")

    # Verify calculations
    assert processed.tokens_removed == 8
    assert processed.token_reduction_ratio == 0.6
    assert not processed.has_critical_loss
    assert not processed.has_high_priority_loss

    print("\n✅ Modification case works correctly")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: ProcessedPrompt with CRITICAL loss
print("\n" + "=" * 80)
print("TEST 4: ProcessedPrompt with CRITICAL content loss")
print("=" * 80)

try:
    # Simulate critical content removal
    removed_critical = [
        TokenClassification(
            token="1girl",
            priority=PromptTokenPriority.CRITICAL,
            category=None,
            is_quality_tag=False,
        ),
        TokenClassification(
            token="fellatio",
            priority=PromptTokenPriority.HIGH,
            category=NSFWCategory.ORAL,
            is_quality_tag=False,
        ),
    ]

    processed = ProcessedPrompt(
        original="1girl, fellatio, cum, masterpiece, detailed, sharp focus",
        final="masterpiece, detailed",
        original_token_count=15,
        final_token_count=5,
        was_modified=True,
        removed_tokens=removed_critical,
        architecture="SDXL",
        quality_level="high",
    )

    processed.add_warning("Critical character information was removed")
    processed.add_warning("NSFW content was removed")

    print(f"Has critical loss: {processed.has_critical_loss}")
    print(f"Has high priority loss: {processed.has_high_priority_loss}")

    print("\nUser message:")
    print(processed.get_user_message())

    print(f"\nShort summary: {processed.get_short_summary()}")

    assert processed.has_critical_loss
    assert processed.has_high_priority_loss
    assert "CRITICAL" in processed.get_short_summary()
    assert "WARNING" in processed.get_user_message()

    print("\n✅ Critical loss detection works correctly")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: to_dict() serialization
print("\n" + "=" * 80)
print("TEST 5: Serialization with to_dict()")
print("=" * 80)

try:
    processed = ProcessedPrompt(
        original="test prompt",
        final="test",
        original_token_count=5,
        final_token_count=3,
        was_modified=True,
        architecture="SDXL",
        quality_level="balanced",
    )

    data = processed.to_dict()

    print("Serialized keys:")
    for key in sorted(data.keys()):
        print(f"  • {key}: {type(data[key]).__name__}")

    assert "original" in data
    assert "final" in data
    assert "was_modified" in data
    assert "has_critical_loss" in data
    assert "token_reduction_ratio" in data

    print("\n✅ Serialization works correctly")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED")
print("=" * 80)
