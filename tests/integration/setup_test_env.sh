#!/bin/bash
# Setup test environment for integration tests using uv

set -e

echo "🔧 Setting up ML_LIB Integration Test Environment"
echo "=================================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

echo "✅ uv found: $(uv --version)"
echo ""

# Create test environment if needed
TEST_ENV_DIR=".venv-test"

if [ ! -d "$TEST_ENV_DIR" ]; then
    echo "📦 Creating test virtual environment..."
    uv venv "$TEST_ENV_DIR" --python 3.11
fi

echo "✅ Test environment ready: $TEST_ENV_DIR"
echo ""

# Install dependencies
echo "📦 Installing test dependencies..."
echo ""

# Core dependencies
echo "  Installing PyTorch..."
uv pip install --python "$TEST_ENV_DIR" torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "  Installing diffusers and transformers..."
uv pip install --python "$TEST_ENV_DIR" diffusers transformers accelerate

echo "  Installing image processing..."
uv pip install --python "$TEST_ENV_DIR" pillow safetensors

echo "  Installing test framework..."
uv pip install --python "$TEST_ENV_DIR" pytest pytest-timeout

echo "  Installing ml_lib in editable mode..."
cd /src/perpetuocambio/ml_lib
uv pip install --python "tests/integration/$TEST_ENV_DIR" -e .

echo ""
echo "=================================================="
echo "✅ Test environment setup complete!"
echo ""
echo "🚀 To activate the environment:"
echo "   source tests/integration/$TEST_ENV_DIR/bin/activate"
echo ""
echo "🧪 To run tests:"
echo "   cd tests/integration"
echo "   ./run_tests_uv.sh quick"
echo ""
