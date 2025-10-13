#!/bin/bash
# Integration test runner using uv for ml_lib diffusion module

set -e

echo "🧪 ML_LIB Diffusion Integration Tests (uv)"
echo "==========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Run ./setup_test_env.sh first"
    exit 1
fi

# Check dependencies
echo "📦 Checking dependencies..."
uv run --no-project python -c "import torch; print('✅ torch installed')" 2>/dev/null || {
    echo "⚠️  torch not installed. Run ./setup_test_env.sh"
    exit 1
}
uv run --no-project python -c "import diffusers; print('✅ diffusers installed')" 2>/dev/null || {
    echo "⚠️  diffusers not installed. Run ./setup_test_env.sh"
    exit 1
}

# Check if Ollama is running
echo "🤖 Checking Ollama..."
if curl -s http://localhost:11434/api/version &> /dev/null; then
    echo "✅ Ollama is running"
    OLLAMA_AVAILABLE=true
else
    echo "⚠️  Ollama not detected (tests with Ollama will be skipped)"
    OLLAMA_AVAILABLE=false
fi

# Check CUDA availability
echo "🎮 Checking GPU..."
uv run --no-project python -c "import torch; print('✅ CUDA available' if torch.cuda.is_available() else '⚠️  CUDA not available (will use CPU - slow)')" 2>/dev/null

echo ""
echo "==========================================="
echo ""

# Parse command line arguments
TEST_SUITE="${1:-quick}"

cd "$(dirname "$0")"

case "$TEST_SUITE" in
    "basic")
        echo "🏃 Running basic generation tests..."
        uv run pytest test_image_generation.py::TestBasicGeneration -v -s
        ;;

    "character")
        echo "🏃 Running character generation tests..."
        uv run pytest test_character_generation.py -v -s
        ;;

    "nsfw")
        echo "🏃 Running NSFW content tests..."
        echo "⚠️  Running adult content tests - output contains NSFW material"
        uv run pytest test_adult_content_generation.py -v -s -m nsfw
        ;;

    "ollama")
        if [ "$OLLAMA_AVAILABLE" = true ]; then
            echo "🏃 Running tests with Ollama integration..."
            uv run pytest test_image_generation.py::TestIntelligentPipeline -v -s
        else
            echo "❌ Ollama not available. Skipping Ollama tests."
            exit 1
        fi
        ;;

    "quick")
        echo "🏃 Running quick smoke test..."
        echo "  This will generate 1 test image (512x512, ~2-3s on GPU)"
        uv run pytest test_image_generation.py::TestBasicGeneration::test_simple_generation -v -s
        ;;

    "all")
        echo "🏃 Running ALL integration tests..."
        echo "⚠️  This will take a while and generate many images"
        uv run pytest -v -s
        ;;

    *)
        echo "❌ Unknown test suite: $TEST_SUITE"
        echo ""
        echo "Usage: $0 [suite]"
        echo ""
        echo "Available test suites:"
        echo "  quick      - Quick smoke test (1 image) [DEFAULT]"
        echo "  basic      - Basic image generation tests"
        echo "  character  - Character generation tests"
        echo "  nsfw       - NSFW/adult content tests"
        echo "  ollama     - Tests with Ollama semantic analysis"
        echo "  all        - All integration tests"
        echo ""
        echo "Examples:"
        echo "  ./run_tests_uv.sh          # Quick test"
        echo "  ./run_tests_uv.sh basic    # Basic tests"
        echo "  ./run_tests_uv.sh nsfw     # Adult content tests"
        exit 1
        ;;
esac

echo ""
echo "==========================================="
echo "✅ Tests completed!"
echo ""
echo "📁 Generated images are in temporary pytest directories"
echo "   Check the test output for exact paths"
echo ""
