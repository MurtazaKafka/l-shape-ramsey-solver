#!/bin/bash
# Run script for L-shape Ramsey solver on GPU server

# Print banner
echo "========================================================"
echo "       L-shape Ramsey Solver with Llama 3.3 70B         "
echo "========================================================"

# Set strict error handling
set -e

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "Found NVIDIA GPU:"
    nvidia-smi -L
    echo ""
    echo "GPU Memory Usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv
    echo ""
else
    echo "WARNING: No NVIDIA GPU detected. This will be extremely slow."
    echo "Consider running on a machine with a GPU."
    read -p "Continue anyway? (y/n): " continue_without_gpu
    if [[ "$continue_without_gpu" != "y" ]]; then
        echo "Exiting."
        exit 1
    fi
fi

# Check for required python packages
echo "Checking for required Python packages..."
missing_packages=0

# Function to check if a Python package is installed
check_package() {
    if python -c "import $1" &> /dev/null; then
        echo "✓ $1 is installed."
    else
        echo "✗ $1 is not installed. Run './install_requirements.sh' first."
        missing_packages=$((missing_packages + 1))
    fi
}

check_package torch
check_package transformers
check_package numpy
check_package matplotlib
check_package bitsandbytes

if [[ $missing_packages -gt 0 ]]; then
    echo "Please install missing packages before continuing."
    echo "Run: ./install_requirements.sh"
    exit 1
fi

# Find model locations
echo "Looking for Llama model files..."
python search_model.py

# Ask user for model path
read -p "Enter the path to the Llama model directory: " model_path

# Update model paths in scripts
echo "Updating model paths in scripts..."
python update_model_path.py "$model_path"

# Test model loading
echo "Testing model loading..."
python test_model_loading.py

# If the test passed, continue
if [[ $? -eq 0 ]]; then
    echo "Model loading test passed!"
    
    # Ask user which grid sizes to solve
    echo ""
    echo "Which grid sizes do you want to solve?"
    echo "1) Small grid (3x3 only)"
    echo "2) Medium grids (3x3 to 6x6)"
    echo "3) Large grids (3x3 to 10x10)"
    echo "4) Custom range"
    read -p "Enter your choice (1-4): " grid_choice
    
    case $grid_choice in
        1)
            grid_param="--grid-size 3"
            ;;
        2)
            grid_param="--min-grid-size 3 --max-grid-size 6"
            ;;
        3)
            grid_param="--min-grid-size 3 --max-grid-size 10"
            ;;
        4)
            read -p "Enter minimum grid size: " min_size
            read -p "Enter maximum grid size: " max_size
            grid_param="--min-grid-size $min_size --max-grid-size $max_size"
            ;;
        *)
            echo "Invalid choice. Using default (3x3 only)."
            grid_param="--grid-size 3"
            ;;
    esac
    
    # Ask for number of iterations
    read -p "Enter number of iterations per island (default: 5): " iterations
    iterations=${iterations:-5}
    
    # Ask for time limit
    read -p "Enter time limit in seconds (default: 300): " time_limit
    time_limit=${time_limit:-300}
    
    # Show final command
    echo ""
    echo "Running command:"
    echo "python llama_funsearch.py $grid_param --iterations $iterations --time-limit $time_limit"
    echo ""
    
    # Run the main script
    python llama_funsearch.py $grid_param --iterations $iterations --time-limit $time_limit
    
    # At the end, list the results
    echo ""
    echo "Results are saved in the funsearch_results directory."
    if [[ -d "funsearch_results" ]]; then
        echo "Found solutions for grid sizes:"
        for dir in funsearch_results/grid_*; do
            if [[ -d "$dir" ]]; then
                grid_size=$(basename "$dir" | cut -d'_' -f2)
                num_solutions=$(ls -1 "$dir"/solution_* 2>/dev/null | wc -l)
                if [[ $num_solutions -gt 0 ]]; then
                    echo "  Grid $grid_size: $num_solutions solution(s)"
                fi
            fi
        done
    fi
else
    echo "Model loading test failed. Please check the error messages above."
    exit 1
fi 