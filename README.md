# L-Shape Ramsey Problem Solver with LLM-FunSearch

## This README.md file need to be updated!!









This repository contains an implementation of the FunSearch algorithm using a locally installed Llama model (specifically Llama 3.3 70B) to solve the L-shape Ramsey problem for grid sizes ranging from 3x3 up to 20x20.

## Problem Description

The L-shape Ramsey problem asks for a 3-coloring of an n×n grid such that no L-shape is monochromatic (contains all the same color). An L-shape consists of three points where two points are equidistant from the third point, forming a right angle.

Example L-shapes:
- Points at (0,0), (2,0), and (2,2) form an L-shape
- Points at (1,1), (1,3), and (3,3) form an L-shape

## Setup Instructions

### 1. Install Requirements

Run the installation script to install all required dependencies:

```bash
chmod +x install_requirements.sh
./install_requirements.sh
```

This will install PyTorch with CUDA support, Transformers, and other necessary libraries.

### 2. Test Model Loading

Before running the full FunSearch, it's a good idea to test if the model can be loaded correctly:

```bash
python test_model_loading.py --model-path /path/to/model
```

For a more comprehensive test including actually loading the model and testing generation:

```bash
python test_model_loading.py --model-path /path/to/model --load-model
```

### 3. Run FunSearch

To run the FunSearch algorithm for a specific grid size:

```bash
python llama_funsearch.py --grid-size 3
```

To run for multiple grid sizes (from min to max):

```bash
python llama_funsearch.py --min-grid-size 3 --max-grid-size 8
```

Additional options:
- `--iterations`: Number of iterations per island (default: 5)
- `--time-limit`: Time limit in seconds (default: 300)
- `--model-path`: Path to the model directory (default: standard location)
- `--temperature`: Temperature for generation (default: 0.7)

## File Structure

- `llama_funsearch.py`: Main implementation of the FunSearch algorithm
- `l_shape_ramsey.py`: Implementation of the L-shape Ramsey problem
- `test_model_loading.py`: Script to test if the model can be loaded correctly
- `install_requirements.sh`: Script to install required dependencies

## Results

Results, including valid solutions and visualizations, will be saved in the `funsearch_results` directory, organized by grid size.

## GPU Requirements

The script is designed to run on an NVIDIA A6000 GPU (48GB VRAM) or similar. It uses 8-bit or 4-bit quantization to fit the 70B parameter model in memory. If you have less VRAM, you might need to use a smaller model or further adjust quantization settings.
