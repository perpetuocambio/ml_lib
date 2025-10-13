"""Simple test to verify Ollama integration works."""

import logging
logging.basicConfig(level=logging.INFO)

from ml_lib.diffusion.services.ollama_selector import OllamaModelSelector

print("=" * 80)
print("TESTING OLLAMA INTEGRATION")
print("=" * 80)
print()

# Simple prompt
PROMPT = "an old female elephant in the forest, anthro, realistic"

try:
    print("1. Initializing OllamaModelSelector...")
    selector = OllamaModelSelector(
        ollama_model="dolphin3",
        ollama_url="http://localhost:11434",
        auto_manage_server=True,  # Auto-start Ollama
    )
    print("✅ Selector initialized")
    print()

    print("2. Analyzing prompt...")
    print(f"Prompt: {PROMPT}")
    print()

    analysis = selector.analyze_prompt(PROMPT)

    if analysis:
        print("✅ Analysis successful!")
        print(f"Style: {analysis.style} (confidence: {analysis.style_confidence:.2f})")
        print(f"Content: {analysis.content_type} (confidence: {analysis.content_confidence:.2f})")
        print(f"Suggested model: {analysis.suggested_base_model}")
        print(f"Key concepts: {analysis.key_concepts}")
        print(f"Recommended LoRA tags: {analysis.recommended_lora_tags}")
        print(f"Suggested steps: {analysis.suggested_steps}")
        print(f"Suggested CFG: {analysis.suggested_cfg}")
    else:
        print("❌ Analysis returned None - Ollama might not be working")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
