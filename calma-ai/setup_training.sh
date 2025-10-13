#!/bin/bash
# Setup script for Calma AI training environment

set -e  # Exit on error

echo "================================================================================"
echo "Calma AI Training Environment Setup"
echo "================================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "train_with_hf_dataset.py" ]; then
    echo -e "${RED}Error: Please run this script from the calma-ai directory${NC}"
    exit 1
fi

# Check for virtual environment
if [ -d "calma" ]; then
    echo -e "${GREEN}✓${NC} Found existing virtual environment: calma"
    VENV_DIR="calma"
elif [ -d "venv" ]; then
    echo -e "${GREEN}✓${NC} Found existing virtual environment: venv"
    VENV_DIR="venv"
else
    echo -e "${YELLOW}⚠${NC} No virtual environment found. Creating one..."
    python3 -m venv calma
    VENV_DIR="calma"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source ${VENV_DIR}/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing training requirements..."
echo "This may take several minutes..."
pip install -r requirements-training.txt

# Check if installation was successful
echo ""
echo "Verifying installation..."
python3 -c "import torch; import transformers; import datasets; import peft; print('✓ All core packages installed')"

# Run test script
echo ""
echo "================================================================================"
echo "Running setup tests..."
echo "================================================================================"
echo ""
python3 test_setup.py

TEST_RESULT=$?

echo ""
echo "================================================================================"
echo "Setup Complete"
echo "================================================================================"
echo ""

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Login to Hugging Face (if not already):"
    echo "   huggingface-cli login"
    echo ""
    echo "2. Test with a small sample (quick, ~5-10 minutes):"
    echo "   python3 train_with_hf_dataset.py --max-samples 100 --epochs 1"
    echo ""
    echo "3. Run full training (slow, ~2-4 hours depending on hardware):"
    echo "   python3 train_with_hf_dataset.py"
    echo ""
    echo "For more options, see: RETRAINING_GUIDE.md"
else
    echo -e "${YELLOW}⚠ Some tests failed.${NC}"
    echo ""
    echo "If Hugging Face authentication failed:"
    echo "  huggingface-cli login"
    echo ""
    echo "Then re-run the tests:"
    echo "  python3 test_setup.py"
fi

echo ""
echo "Don't forget to activate the virtual environment before training:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
