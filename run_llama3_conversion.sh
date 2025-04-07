#!/bin/bash
# Script to convert Llama 3 model and run FunSearch

# Set error handling
set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print banner
echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}   Llama 3 Model Conversion and FunSearch   ${NC}"
echo -e "${GREEN}=============================================${NC}"

# Default values
INPUT_DIR="/home/DAVIDSON/munikzad/.llama/checkpoints/Llama3.3-70B-Instruct"
OUTPUT_DIR="./llama3_hf"
MODEL_SIZE="Llama-3-70B"
NUM_SHARDS=8
GRID_SIZE=3
COLORS=3

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --input-dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model-size)
      MODEL_SIZE="$2"
      shift 2
      ;;
    --num-shards)
      NUM_SHARDS="$2"
      shift 2
      ;;
    --grid-size)
      GRID_SIZE="$2"
      shift 2
      ;;
    --colors)
      COLORS="$2"
      shift 2
      ;;
    --install-deps)
      INSTALL_DEPS=true
      shift
      ;;
    --skip-conversion)
      SKIP_CONVERSION=true
      shift
      ;;
    --skip-update)
      SKIP_UPDATE=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --input-dir DIR       Input directory with Llama 3 model files (default: $INPUT_DIR)"
      echo "  --output-dir DIR      Output directory for converted model (default: $OUTPUT_DIR)"
      echo "  --model-size SIZE     Model size (default: $MODEL_SIZE)"
      echo "  --num-shards N        Number of model shards (default: $NUM_SHARDS)"
      echo "  --grid-size N         Grid size for FunSearch (default: $GRID_SIZE)"
      echo "  --colors N            Number of colors for FunSearch (default: $COLORS)"
      echo "  --install-deps        Install dependencies"
      echo "  --skip-conversion     Skip model conversion"
      echo "  --skip-update         Skip updating llama_funsearch.py"
      echo "  --help                Display this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Step 1: Install dependencies if requested
if [ "$INSTALL_DEPS" = true ]; then
  echo -e "\n${YELLOW}[1/4] Installing dependencies...${NC}"
  ./install_requirements.sh
else
  echo -e "\n${YELLOW}[1/4] Skipping dependency installation...${NC}"
fi

# Step 2: Convert model
if [ "$SKIP_CONVERSION" != true ]; then
  echo -e "\n${YELLOW}[2/4] Converting Llama 3 model...${NC}"
  
  # Check if input directory exists
  if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}Error: Input directory $INPUT_DIR does not exist.${NC}"
    exit 1
  fi
  
  # Create output directory if it doesn't exist
  mkdir -p "$OUTPUT_DIR"
  
  # Run conversion script
  echo -e "${YELLOW}Converting model from $INPUT_DIR to $OUTPUT_DIR...${NC}"
  echo -e "${YELLOW}This may take a while depending on your system.${NC}"
  ./convert_llama3_to_hf.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --model_size "$MODEL_SIZE" --num_shards "$NUM_SHARDS"
  
  # Check if conversion was successful
  if [ ! -f "$OUTPUT_DIR/config.json" ]; then
    echo -e "${RED}Error: Conversion failed. config.json not found in $OUTPUT_DIR${NC}"
    exit 1
  fi
  
  echo -e "${GREEN}Model conversion completed successfully!${NC}"
else
  echo -e "\n${YELLOW}[2/4] Skipping model conversion...${NC}"
  
  # Check if output directory exists with converted model
  if [ ! -f "$OUTPUT_DIR/config.json" ]; then
    echo -e "${RED}Warning: $OUTPUT_DIR/config.json not found. Model conversion may be needed.${NC}"
    echo -e "${YELLOW}Continuing anyway...${NC}"
  fi
fi

# Step 3: Update llama_funsearch.py
if [ "$SKIP_UPDATE" != true ]; then
  echo -e "\n${YELLOW}[3/4] Updating llama_funsearch.py...${NC}"
  
  # Check if llama_funsearch.py exists
  if [ ! -f "llama_funsearch.py" ]; then
    echo -e "${RED}Error: llama_funsearch.py not found${NC}"
    exit 1
  fi
  
  # Run update script
  ./update_model_path.py --file llama_funsearch.py --model_path "$OUTPUT_DIR"
else
  echo -e "\n${YELLOW}[3/4] Skipping llama_funsearch.py update...${NC}"
fi

# Step 4: Run FunSearch
echo -e "\n${YELLOW}[4/4] Running FunSearch...${NC}"
echo -e "${YELLOW}Running with grid size $GRID_SIZE and $COLORS colors...${NC}"
python llama_funsearch.py --grid_size "$GRID_SIZE" --colors "$COLORS" --model_path "$OUTPUT_DIR"

echo -e "\n${GREEN}=============================================${NC}"
echo -e "${GREEN}   Process completed!   ${NC}"
echo -e "${GREEN}=============================================${NC}" 