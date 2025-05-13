#!/bin/bash
# Script to fix GPU loading for the Llama 3.3 70B model and run FunSearch

# Set error handling
set -e

# Define colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}   L-Shape Ramsey with Llama 3.3 on GPU   ${NC}"
echo -e "${GREEN}==============================================${NC}"

# Check if bitsandbytes is installed (required for 4-bit quantization)
if ! python -c "import bitsandbytes" &> /dev/null; then
  echo -e "${YELLOW}Installing bitsandbytes for GPU quantization...${NC}"
  pip install bitsandbytes
else
  echo -e "${GREEN}bitsandbytes already installed.${NC}"
fi

# Create offload folder if it doesn't exist
mkdir -p offload

# Step 1: Check if the model has been converted
if [ ! -f "./llama3_hf/config.json" ]; then
  echo -e "${RED}Error: Converted model not found in ./llama3_hf${NC}"
  echo -e "${YELLOW}Please run the model conversion first with: ./run_llama3_conversion.sh${NC}"
  exit 1
fi

# Step 2: Update the model loading code to better handle GPU memory
echo -e "\n${YELLOW}Fixing model loading code for better GPU memory management...${NC}"
./fix_gpu_loading.py

# Step 3: Update the model path
echo -e "\n${YELLOW}Updating model path in llama_funsearch.py...${NC}"
sed -i 's|self\.model_path = model_path or "[^"]*"|self.model_path = model_path or "./llama3_hf"|g' llama_funsearch.py

# Check if update was successful
if ! grep -q './llama3_hf' llama_funsearch.py; then
  echo -e "${YELLOW}Could not update model path automatically, trying alternative approach...${NC}"
  sed -i 's|model_path = model_path or .*$|model_path = model_path or "./llama3_hf"|g' llama_funsearch.py
  
  if ! grep -q './llama3_hf' llama_funsearch.py; then
    echo -e "${RED}Warning: Could not update model path automatically.${NC}"
    echo -e "${YELLOW}The script will use the model path specified on the command line.${NC}"
  fi
fi

# Step 4: Run FunSearch with additional memory management settings
echo -e "\n${YELLOW}Running FunSearch with GPU optimizations...${NC}"

# Set environment variables for better GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0

# Run with command-line arguments
echo -e "${YELLOW}Running with grid size 3...${NC}"
python llama_funsearch.py --grid-size 3 --min-grid-size 3 --max-grid-size 3 --model-path ./llama3_hf

echo -e "\n${GREEN}==============================================${NC}"
echo -e "${GREEN}   Process completed!   ${NC}"
echo -e "${GREEN}==============================================${NC}" 