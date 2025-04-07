#!/bin/bash
# Script to install required packages for Llama 3 model conversion and usage

# Set error handling
set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print banner
echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}   Installing requirements for Llama 3 Model   ${NC}"
echo -e "${GREEN}=============================================${NC}"

# Install basic requirements
echo -e "\n${YELLOW}Installing basic requirements...${NC}"
pip install torch transformers accelerate tqdm

# Install requirements for model conversion
echo -e "\n${YELLOW}Installing requirements for model conversion...${NC}"
pip install protobuf safetensors

# Try to install sentencepiece without building from source
echo -e "\n${YELLOW}Attempting to install sentencepiece via pip...${NC}"
pip install --only-binary :all: sentencepiece

# Check if sentencepiece installed successfully
if python -c "import sentencepiece" &> /dev/null; then
    echo -e "${GREEN}sentencepiece installed successfully!${NC}"
else
    echo -e "${YELLOW}Failed to install sentencepiece via pip. Trying alternative approach...${NC}"
    
    # Check if conda is available
    if command -v conda &> /dev/null; then
        echo -e "${YELLOW}Using conda to install sentencepiece...${NC}"
        conda install -c conda-forge sentencepiece -y
    else
        echo -e "${RED}Warning: sentencepiece installation failed and conda is not available.${NC}"
        echo -e "${YELLOW}The conversion script will try to copy the tokenizer manually.${NC}"
    fi
fi

# Install blobfile for tokenizer conversion
echo -e "\n${YELLOW}Installing blobfile for tokenizer conversion...${NC}"
pip install blobfile

echo -e "\n${GREEN}=============================================${NC}"
echo -e "${GREEN}   Requirements installation completed!   ${NC}"
echo -e "${GREEN}=============================================${NC}"
echo -e "\nNext steps:"
echo -e "1. Convert the model using: ./convert_llama3_to_hf.py --input_dir ... --output_dir ..."
echo -e "2. Update the model path using: ./update_model_path.py --file llama_funsearch.py --model_path ..."
echo -e "3. Run the FunSearch algorithm: python llama_funsearch.py --grid_size 3 --colors 3" 