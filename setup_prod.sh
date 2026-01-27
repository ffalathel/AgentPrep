#!/bin/bash
# Production setup script for AgentPrep
# This script creates a clean virtual environment with correct dependencies

set -e  # Exit on error

echo "=========================================="
echo "AgentPrep Production Setup"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Verify Python 3.8+
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "❌ Error: Python 3.8+ is required. Found: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
VENV_NAME="${1:-venv}"
if [ -d "$VENV_NAME" ]; then
    echo "⚠️  Virtual environment '$VENV_NAME' already exists."
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_NAME"
    else
        echo "Using existing virtual environment."
    fi
fi

if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment: $VENV_NAME"
    python3 -m venv "$VENV_NAME"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
# Install package in production mode
echo "Installing agentprep package..."
pip install .

# Note: If you want development dependencies, use:
# pip install -e ".[dev]"

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import pandas; import numpy; import pydantic; import yaml; import pyarrow; print('✓ All core dependencies installed successfully')" || {
    echo "❌ Error: Dependency verification failed"
    exit 1
}

# Check versions
echo ""
echo "Installed versions:"
python3 -c "import pandas; import numpy; print(f'  pandas: {pandas.__version__}')"
python3 -c "import numpy; print(f'  numpy: {numpy.__version__}')"
python3 -c "import pydantic; print(f'  pydantic: {pydantic.__version__}')"

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
