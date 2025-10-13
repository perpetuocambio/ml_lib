#!/bin/bash
# Integration test runner for ml_lib diffusion module

set -e

echo "🧪 ML_LIB Diffusion Integration Tests"
echo "======================================"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Installing..."
    pip install pytest pytest-timeout
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import torch; print('✅ torch installed')" 2>/dev/null || echo "⚠️  torch not installed (some tests will be skipped)"
python3 -c "import diffusers; print('✅ diffusers installed')" 2>/dev/null || echo "⚠️  diffusers not installed (tests will fail)"
python3 -c "import transformers; print('✅ transformers installed')" 2>/dev/null || echo "⚠️  transformers not installed (tests will fail)"

# Check if Ollama is running
echo ""
echo "🤖 Checking Ollama..."
if curl -s http://localhost:11434/api/version &> /dev/null; then
    echo "✅ Ollama is running"
else
    echo "⚠️  Ollama not detected (tests with Ollama will be skipped)"
fi

# Check CUDA availability
echo ""
echo "🎮 Checking GPU..."
python3 -c "import torch; print('✅ CUDA available' if torch.cuda.is_available() else '⚠️  CUDA not available (will use CPU - slow)')" 2>/dev/null || echo "⚠️  Cannot check CUDA"

echo ""
echo "========================================"
echo ""

# Parse command line arguments
TEST_SUITE="all"
if [ "$1" != "" ]; then
    TEST_SUITE="$1"
fi

cd "$(dirname "$0")"

case "$TEST_SUITE" in
    "basic")
        echo "🏃 Running basic generation tests..."
        pytest test_image_generation.py::TestBasicGeneration -v -s
        ;;

    "character")
        echo "🏃 Running character generation tests..."
        pytest test_character_generation.py -v -s
        ;;

    "nsfw")
        echo "🏃 Running NSFW content tests..."
        echo "⚠️  Running adult content tests - output contains NSFW material"
        pytest test_adult_content_generation.py -v -s -m nsfw
        ;;

    "ollama")
        echo "🏃 Running tests with Ollama integration..."
        pytest test_image_generation.py::TestIntelligentPipeline -v -s
        ;;

    "quick")
        echo "🏃 Running quick smoke tests..."
        pytest test_image_generation.py::TestBasicGeneration::test_simple_generation -v -s
        ;;

    "all")
        echo "🏃 Running ALL integration tests..."
        echo "⚠️  This will take a while and generate many images"
        pytest -v -s
        ;;

    *)
        echo "❌ Unknown test suite: $TEST_SUITE"
        echo ""
        echo "Usage: $0 [suite]"
        echo ""
        echo "Available test suites:"
        echo "  basic      - Basic image generation tests"
        echo "  character  - Character generation tests"
        echo "  nsfw       - NSFW/adult content tests"
        echo "  ollama     - Tests with Ollama semantic analysis"
        echo "  quick      - Quick smoke test"
        echo "  all        - All integration tests (default)"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "✅ Tests completed!"
echo ""
echo "📁 Generated images are in: /tmp/pytest-*/test_*/"
echo ""
