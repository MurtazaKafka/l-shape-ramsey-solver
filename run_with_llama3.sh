#!/bin/bash
# Script to run the L-shape Ramsey solver with Llama 3.3 70B

# Set error handling
set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print banner
echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}   L-shape Ramsey Solver with Llama 3.3 70B   ${NC}"
echo -e "${GREEN}=============================================${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
    exit 1
fi

# Step 1: Install dependencies
echo -e "\n${YELLOW}[1/4] Installing dependencies...${NC}"
python3 -m pip install torch transformers sentencepiece blobfile protobuf accelerate

# Step 2: Convert the model
echo -e "\n${YELLOW}[2/4] Converting the Llama model to Hugging Face format...${NC}"
echo -e "${YELLOW}This may take a while depending on your system.${NC}"

input_dir="/home/DAVIDSON/munikzad/.llama/checkpoints/Llama3.3-70B-Instruct"
output_dir="./llama3_hf"

# Check if the input directory exists
if [ ! -d "$input_dir" ]; then
    echo -e "${RED}Error: Input directory $input_dir does not exist.${NC}"
    echo -e "${YELLOW}Please update the input_dir variable in this script.${NC}"
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Run the conversion script
python3 convert_meta_to_hf.py --input_dir "$input_dir" --output_dir "$output_dir" --skip_deps

# Check if conversion was successful
if [ ! -f "$output_dir/config.json" ]; then
    echo -e "${RED}Error: Conversion failed. config.json not found in $output_dir${NC}"
    echo -e "${YELLOW}Please check the error messages above and try again.${NC}"
    exit 1
fi

# Step 3: Update the model path in llama_funsearch.py
echo -e "\n${YELLOW}[3/4] Updating the model path in llama_funsearch.py...${NC}"
python3 update_model_path.py --file llama_funsearch.py --model_path "$output_dir"

# Step 4: Run the FunSearch algorithm
echo -e "\n${YELLOW}[4/4] Running FunSearch with the converted model...${NC}"
python3 llama_funsearch.py --grid_size 3 --colors 3

echo -e "\n${GREEN}=============================================${NC}"
echo -e "${GREEN}   Completed!   ${NC}"
echo -e "${GREEN}=============================================${NC}" 