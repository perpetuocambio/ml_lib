"""Test Ollama prompt analysis."""

import logging

logging.basicConfig(level=logging.INFO)

from ml_lib.diffusion.prompt.ollama_selector import OllamaModelSelector

PROMPT = """an old female elephant in the forest, anthro, blue eyes, orgasm face"""

print("Testing Ollama analysis...")
print(f"Prompt: {PROMPT}")
print()

selector = OllamaModelSelector(ollama_model="dolphin3")
analysis = selector.analyze_prompt(PROMPT)

if analysis:
    print(f"Style: {analysis.style}")
    print(f"Content type: {analysis.content_type}")
    print(f"Suggested base model: {analysis.suggested_base_model}")
    print(f"Key concepts: {analysis.key_concepts}")
    print(f"Trigger words: {analysis.trigger_words}")
    print(f"Recommended LoRA tags: {analysis.recommended_lora_tags}")
else:
    print("Analysis failed!")
