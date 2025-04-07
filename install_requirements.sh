#!/bin/bash
# Install requirements for running llama_funsearch.py

echo "Installing required packages for llama_funsearch.py"
echo "--------------------------------------------------"

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed. Please install Python and pip first."
    exit 1
fi

# Required packages
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing Transformers and related libraries..."
pip install transformers==4.36.2 accelerate==0.25.0 bitsandbytes==0.41.3 sentencepiece==0.1.99

echo "Installing other dependencies..."
pip install numpy matplotlib

echo "--------------------------------------------------"
echo "Installation complete! You can now run:"
echo "python test_model_loading.py --load-model"
echo "to test if the model can be loaded correctly."
echo "--------------------------------------------------" 