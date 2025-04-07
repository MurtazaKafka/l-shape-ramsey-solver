#!/bin/bash
# Install transformers and related packages

echo "Installing transformers and dependencies for Llama 3.3 70B..."
echo "This will ensure the LLM can be loaded properly."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not available."
    echo "Please check your Python installation."
    exit 1
fi

# Install transformers with dependencies
echo "Installing transformers..."
pip install transformers==4.36.2

echo "Installing accelerate for GPU support..."
pip install accelerate==0.25.0

echo "Installing bitsandbytes for quantization..."
pip install bitsandbytes==0.41.3

echo "Installing sentencepiece for tokenization..."
pip install sentencepiece==0.1.99

echo "Installing other dependencies..."
pip install numpy matplotlib

# Display package versions for verification
echo "Installed package versions:"
pip list | grep -E "transformers|accelerate|bitsandbytes|sentencepiece|torch"

echo "Installation complete!"
echo "You can now run: python check_imports.py"
echo "to verify the installations."

# Check if the transformers module can be imported
echo "Verifying transformers installation..."
if python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; print('Transformers successfully imported')" &> /dev/null; then
    echo "✓ Transformers installation verified!"
else
    echo "✗ There was a problem with the transformers installation."
    echo "Please run: python check_imports.py"
    echo "for more details."
fi 