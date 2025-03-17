# L-Shape Ramsey Problem Solver

This project implements various approaches to solve the L-shape Ramsey problem, which involves finding grid configurations that avoid monochromatic L-shapes while using a limited number of colors.

## Problem Description

The L-shape Ramsey problem asks: Given a grid of size n×n and k colors, what is the largest possible grid that can be colored without creating any monochromatic L-shapes? An L-shape is formed by three cells of the same color in an L configuration.

## Implementation Approaches

1. **Hierarchical Approach** (`l_shape_hierarchical.py`)
   - Builds solutions by combining known valid smaller grids
   - Uses multiple generation strategies
   - Maintains a cache of known solutions

2. **GPU-Accelerated Approach** (`l_shape_gpu.py`)
   - Uses PyTorch for GPU-accelerated computation
   - Implements efficient L-shape detection on GPU
   - Optimized for large grid sizes

3. **FunSearch Approach** (`implementation/`)
   - Uses evolutionary algorithms to find solutions
   - Implements a sandbox for safe code execution
   - Includes visualization tools

## Setup

1. Clone the repository:
```bash
git clone https://github.com/MurtazaKafka/l-shape-ramsey.git
cd l-shape-ramsey
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the hierarchical solver:
```bash
python l_shape_hierarchical.py
```

2. Run the GPU-accelerated solver:
```bash
python l_shape_gpu.py
```

3. Run the FunSearch implementation:
```bash
cd implementation
python run_l_shape_funsearch.py
```

## Project Structure

```
l-shape-ramsey/
├── README.md
├── requirements.txt
├── l_shape_hierarchical.py
├── l_shape_gpu.py
├── l_shape_ramsey.py
├── l_shape_analysis.py
├── l_shape_funsearch.py
└── implementation/
    ├── run_l_shape_funsearch.py
    ├── sampler.py
    ├── evaluator.py
    ├── sandbox.py
    ├── programs_database.py
    ├── funsearch.py
    └── visualizations/
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Other dependencies listed in requirements.txt

## License

MIT License
