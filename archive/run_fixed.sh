#!/bin/bash
# Script to fix syntax error and run FunSearch with GPU optimizations

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

# Step 1: Fix the syntax error in llama_funsearch.py
echo -e "\n${YELLOW}Fixing syntax error in llama_funsearch.py...${NC}"
python fix_syntax_error.py

# Step 2: Restore the llama_funsearch.py from backup if needed
if [ -f "llama_funsearch.py.backup" ]; then
  echo -e "\n${YELLOW}Restoring from backup and updating model path...${NC}"
  cp llama_funsearch.py.backup llama_funsearch.py
  
  # Update the model path using sed with careful pattern matching
  sed -i 's|self\.model_path = model_path or "[^"]*"|self.model_path = model_path or "./llama3_hf"|g' llama_funsearch.py
  
  # Also fix _load_model function with our GPU optimized version
  echo -e "\n${YELLOW}Updating model loading code for better GPU handling...${NC}"
  ./fix_gpu_loading.py
  
  # Fix any syntax errors again just to be safe
  python fix_syntax_error.py
fi

# Step 3: Create offload folder if it doesn't exist
mkdir -p offload

# Step 4: Check if bitsandbytes is installed (required for 4-bit quantization)
if ! python -c "import bitsandbytes" &> /dev/null; then
  echo -e "\n${YELLOW}Installing bitsandbytes for GPU quantization...${NC}"
  pip install bitsandbytes
else
  echo -e "\n${GREEN}bitsandbytes already installed.${NC}"
fi

# Step 5: Run FunSearch with additional memory management settings
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