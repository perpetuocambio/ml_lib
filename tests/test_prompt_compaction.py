"""Test prompt compaction with real NSFW prompts."""

import logging

logging.basicConfig(level=logging.INFO)

# Test 1: Import the compaction module
print("=" * 80)
print("TEST 1: Import modules")
print("=" * 80)

try:
    from ml_lib.diffusion.domain.services.prompt_analyzer import PromptAnalyzer
    from ml_lib.diffusion.domain.services.prompt_compactor import PromptCompactor
    from ml_lib.diffusion.models.content_tags import analyze_nsfw_content
    print("✅ Modules imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Analyze NSFW content
print("\n" + "=" * 80)
print("TEST 2: Analyze NSFW Content")
print("=" * 80)

test_prompt = "score_9, score_8_up, score_7_up, score_6_up, masterpiece, Expressiveh, hyper detail, 1girl, brown hair, straight hair, long hair, perfect body, fit, sexy, light blue eyes, nude, guy face out of frame, warm light, kitchen, ruanyi1221, view between legs fellatio, penis, fellatio, uncensored, from below, testicles, deepthroat, rough sex, cum on mouth"

try:
    analysis = analyze_nsfw_content(test_prompt)
    print(f"Is NSFW: {analysis.is_nsfw}")
    print(f"Confidence: {analysis.confidence:.2f}")
    print(f"Categories: {[cat.value for cat in analysis.categories]}")
    print(f"Recommended LoRA tags: {analysis.recommended_lora_tags[:10]}")
    print("✅ NSFW analysis successful")
except Exception as e:
    print(f"❌ NSFW analysis failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Compact prompt
print("\n" + "=" * 80)
print("TEST 3: Compact Prompt (with transformers)")
print("=" * 80)

analyzer = PromptAnalyzer(use_llm=False)
compactor = PromptCompactor()

try:
    print(f"Original prompt ({len(test_prompt)} chars):")
    print(test_prompt[:150] + "...")
    print()

    result = compactor.compact(test_prompt, preserve_nsfw=True)

    print(f"Compacted prompt ({len(result.compacted_prompt)} chars):")
    print(result.compacted_prompt)
    print()
    print("Metadata:")
    print(f"  Original tokens: {result.original_token_count}")
    print(f"  Final tokens: {result.compacted_token_count}")
    print(f"  Compaction needed: {result.was_compacted}")

    if result.was_compacted:
        print(f"  NSFW preserved: {result.nsfw_content_preserved}")
        print(f"  Core content preserved: {result.core_content_preserved}")
        print(f"  Quality tags removed: {result.quality_tags_removed}")
        print(f"  Removed {len(result.removed_tokens)} tokens")
        if result.removed_tokens:
            print(f"  First removed: {result.removed_tokens[0].token}")

    print("✅ Prompt compaction successful")

except Exception as e:
    print(f"❌ Prompt compaction failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test with longer prompt (should trigger compaction)
print("\n" + "=" * 80)
print("TEST 4: Longer Prompt (should be compacted)")
print("=" * 80)

long_prompt = """masterpiece, best quality, amazing quality, very aesthetic, absurdres, 8k uhd, extremely detailed, RAW photo, professional photography, sharp focus, depth of field, score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, 1girl, brown hair, straight hair, long hair, perfect body, fit, sexy, light blue eyes, nude, naked, breasts, nipples, guy face out of frame, warm light, bedroom interior, view between legs fellatio, penis, fellatio, oral sex, uncensored, from below, testicles, deepthroat, rough sex, cum on mouth, cumshot, facial, bukkake, excessive cum"""

try:
    print(f"Original: {len(long_prompt)} chars")

    result = compactor.compact(long_prompt, preserve_nsfw=True)

    print(f"Compacted: {len(result.compacted_prompt)} chars")
    print()
    print(f"Token reduction: {result.original_token_count} → {result.compacted_token_count}")
    print(f"Compression ratio: {result.compression_ratio:.1%}")
    print()
    print("Compacted prompt:")
    print(result.compacted_prompt)

    # Check what was preserved
    print(f"\n✅ NSFW content preserved: {result.nsfw_content_preserved}")
    print(f"✅ NSFW categories found: {[cat.value for cat in result.nsfw_categories_found]}")

    print("✅ Long prompt compaction successful")

except Exception as e:
    print(f"❌ Long prompt compaction failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test full optimization pipeline
print("\n" + "=" * 80)
print("TEST 5: Full optimization pipeline (optimize_for_model)")
print("=" * 80)

simple_prompt = "1girl, fellatio, cum on face, nude, breasts"

try:
    print(f"Input prompt: {simple_prompt}")

    # First optimize for model
    optimized_positive, optimized_negative = analyzer.optimize_for_model(
        prompt=simple_prompt,
        negative_prompt="low quality, worst quality",
        base_model_architecture="sdxl",
        quality="high"
    )

    # Then compact if needed
    result = compactor.compact(optimized_positive, preserve_nsfw=True)

    print(f"\nOptimized positive (after compaction):")
    print(result.compacted_prompt)
    print(f"\nOptimized negative:")
    print(optimized_negative)
    print(f"\nCompaction stats: {result.original_token_count} → {result.compacted_token_count} tokens ({result.compression_ratio:.1%} kept)")

    print("\n✅ Full optimization successful")

except Exception as e:
    print(f"❌ Full optimization failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED")
print("=" * 80)
