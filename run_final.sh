#!/bin/bash
# Simple script to run llama_funsearch.py with the converted model

# Set error handling
set -e

# Define colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}   Running L-Shape Ramsey FunSearch   ${NC}"
echo -e "${GREEN}==============================================${NC}"

# Check if the model is converted
if [ ! -f "./llama3_hf/config.json" ]; then
  echo -e "${RED}Error: Converted model not found in ./llama3_hf${NC}"
  echo -e "${YELLOW}Please run the model conversion first with: ./run_llama3_conversion.sh${NC}"
  exit 1
fi

# Update llama_funsearch.py manually to use the converted model
echo -e "${YELLOW}Updating llama_funsearch.py to use the converted model...${NC}"

# Make a backup of the original file
cp llama_funsearch.py llama_funsearch.py.backup

# Replace the model path in llama_funsearch.py using sed
sed -i 's|self\.model_path = model_path or "[^"]*"|self.model_path = model_path or "./llama3_hf"|g' llama_funsearch.py

# Check if the replacement worked
if grep -q './llama3_hf' llama_funsearch.py; then
  echo -e "${GREEN}Successfully updated model path in llama_funsearch.py${NC}"
else
  echo -e "${YELLOW}Could not update model path automatically, trying alternative approach...${NC}"
  # Try an alternative approach with a more generic pattern
  sed -i 's|model_path = model_path or .*$|model_path = model_path or "./llama3_hf"|g' llama_funsearch.py
  
  if grep -q './llama3_hf' llama_funsearch.py; then
    echo -e "${GREEN}Successfully updated model path in llama_funsearch.py (alt method)${NC}"
  else
    echo -e "${RED}Could not update model path automatically.${NC}"
    echo -e "${YELLOW}Please edit llama_funsearch.py manually to set self.model_path = \"./llama3_hf\"${NC}"
  fi
fi

# Run the FunSearch algorithm
echo -e "\n${YELLOW}Running FunSearch with the converted model...${NC}"

# Run with proper arguments
python llama_funsearch.py --grid-size 3 --min-grid-size 3 --max-grid-size 3 --model-path ./llama3_hf

echo -e "\n${GREEN}==============================================${NC}"
echo -e "${GREEN}   Process completed!   ${NC}"
echo -e "${GREEN}==============================================${NC}" 