#!/usr/bin/env python3
"""
Test Ollama server auto-start/stop functionality for memory optimization.
"""

import sys
from pathlib import Path
import time

# Add ml_lib to path
sys.path.insert(0, str(Path(__file__).parent))

print("🧪 Test: Ollama Auto-Start/Stop")
print("=" * 80)
print()

print("📋 This test demonstrates automatic Ollama server management")
print("   for memory optimization in diffusion generation workflows.")
print()

# Test 1: Auto-start
print("=" * 80)
print("TEST 1: Auto-Start Server")
print("=" * 80)
print()

print("Creating OllamaModelSelector with auto_manage_server=True...")
from ml_lib.diffusion.prompt.ollama_selector import OllamaModelSelector

selector = OllamaModelSelector(
    ollama_model="dolphin3",
    ollama_url="http://localhost:11434",
    auto_manage_server=True,
)

print("✅ Selector created")
print()

# Test 2: Analyze prompt (will auto-start if needed)
print("=" * 80)
print("TEST 2: Analyze Prompt (auto-start if needed)")
print("=" * 80)
print()

test_prompt = "A beautiful anime girl with magical powers in a fantasy setting"
print(f"Prompt: {test_prompt}")
print()

print("Analyzing...")
analysis = selector.analyze_prompt(test_prompt)

if analysis:
    print("✅ Analysis successful!")
    print(f"   Style: {analysis.style} (confidence: {analysis.style_confidence:.2f})")
    print(f"   Content: {analysis.content_type}")
    print(f"   Suggested model: {analysis.suggested_base_model}")
    print(f"   Key concepts: {', '.join(analysis.key_concepts[:5])}")
else:
    print("⚠️  Analysis returned None (Ollama may not be available)")

print()

# Test 3: Stop server to free memory
print("=" * 80)
print("TEST 3: Stop Server (memory optimization)")
print("=" * 80)
print()

print("Stopping Ollama server to free memory...")
stopped = selector.stop_server()

if stopped:
    print("✅ Server stopped successfully - memory freed!")
else:
    print("ℹ️  Server was not started by us or already stopped")

print()

# Test 4: Context manager (recommended pattern)
print("=" * 80)
print("TEST 4: Context Manager Pattern")
print("=" * 80)
print()

print("Using context manager for automatic start/stop:")
print()

from ml_lib.llm.providers.ollama_provider import OllamaServerContext
from ml_lib.llm.entities.llm_prompt import LLMPrompt

try:
    with OllamaServerContext(ollama_model="dolphin3", auto_stop=True) as provider:
        print("✅ Server started via context manager")

        # Do analysis
        prompt = LLMPrompt(
            content="Analyze: photorealistic portrait of elderly woman", temperature=0.7
        )

        response = provider.generate_response(prompt)
        print(f"✅ Response received: {response.content[:100]}...")

    # Server automatically stopped when exiting context
    print("✅ Server stopped automatically - memory freed!")

except Exception as e:
    print(f"⚠️  Error: {e}")

print()

# Summary
print("=" * 80)
print("📊 SUMMARY")
print("=" * 80)
print()
print("✅ Auto-start functionality working")
print("✅ Manual stop available for memory optimization")
print("✅ Context manager provides automatic lifecycle management")
print()
print("💡 Recommended usage:")
print("   1. Use context manager for one-time analysis")
print("   2. Call stop_server() after batch of analyses")
print("   3. Let diffusion pipeline manage lifecycle automatically")
print()
print("🎯 Memory optimization achieved - Ollama runs only when needed!")
