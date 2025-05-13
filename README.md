# L-Shape Ramsey Problem Solver

![Grid Visualization](https://img.shields.io/badge/Grid-Visualization-brightgreen)
![Llama 3.3 70B](https://img.shields.io/badge/Model-Llama%203.3%2070B-blue)
![GPU Acceleration](https://img.shields.io/badge/GPU-Acceleration-orange)

An advanced implementation to solve the L-shape Ramsey problem using Meta's Llama 3.3 70B and FunSearch algorithms. This project finds valid 3-colorings of n×n grids where no L-shape is monochromatic.

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU with 40GB+ VRAM (for Llama 3.3 70B)
- Conda environment (recommended)

### Installation

```bash
# Clone the repository
git clone <https://github.com/MurtazaKafka/l-shape-ramsey-solver>
cd l-shape-ramsey-solver

# Create and activate conda environment
conda create -n ramsey python=3.10
conda activate ramsey

# Install basic requirements
pip install -r requirements.txt

# Install llama-cpp-python with CUDA support
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" \
FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
```

### Running Simple Tests

```bash
# Test base functionality
python test_visualization.py

# Run deterministic solver for small grids
python fixed_solver.py --grid-size 5
```

### Using FunSearch with Llama

```bash
# Run FunSearch for grid sizes 4-6
python llama_funsearch.py --grid-size 4 --max-grid-size 6 --iterations 5 --time-limit 300
```

## Features

- **Multi-method approach**: Deterministic patterns, FunSearch, and reinforcement learning
- **Scales to large grid sizes**: Solutions for grid sizes up to 10×10 and beyond
- **GPU acceleration**: Uses CUDA for language model and grid operations
- **Visualization tools**: Advanced visualization of grid colorings and L-shapes
- **Extensible framework**: Add your own patterns and algorithms

## Model Files

This repository uses large language models that aren't included directly in the Git repository due to their size. You'll need to download them separately.

### Required Models

- **Llama 3.3 70B (Main model)**: 
  - Location: `models/Llama-3.3-70B-Instruct-Q4_K_M.gguf`
  - Size: ~40GB
  
- **TinyLlama (Lightweight alternative)**:
  - Location: `models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf`
  - Size: ~700MB

### Downloading Models

Use the provided model downloader script:

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Log in to Hugging Face (needed for Llama models)
huggingface-cli login

# Download all models (recommended)
python model_downloader.py

# Or download specific models
python model_downloader.py --model llama3-70b
python model_downloader.py --model tinyllama
```

Models will be saved to the `models/` directory by default.

### Running with Different Models

Specify which model to use:

```bash
# Use the full Llama 3.3 70B model
python llama_funsearch.py --model-path models/Llama-3.3-70B-Instruct-Q4_K_M.gguf --grid-size 5

# Use lighter TinyLlama model for testing or on systems with less VRAM
python llama_funsearch.py --model-path models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf --grid-size 4
```

## Example Solutions

| Grid Size | Visualization | Method |
|-----------|---------------|--------|
| 3×3       | [View](./results/grid_3x3_solution.png) | Latin Square |
| 4×4       | [View](./results/grid_4x4_solution.png) | Specialized Solver |
| 5×5       | [View](./results/grid_5x5_solution.png) | FunSearch |

## Advanced Usage

```bash
# Run with custom parameters
python llama_funsearch.py --temperature 0.8 --grid-size 7 --iterations 10 --time-limit 600

# Use GPU manager to free memory
python gpu_manager.py --info --list --min-free 40000

# Generate comparison visualizations across grid sizes
python llama_funsearch.py --grid-size 3 --max-grid-size 8 --visualize-all
```

## Documentation

For detailed information, see:
- [Full Documentation](./DOCUMENTATION.md)
- [Llama 3.3 Integration](./llama_integration_summary.md)
- [Experiment Results](./llama3_experiment_results.md)

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## References

- [FunSearch Paper](https://arxiv.org/abs/2307.08674)
- [Meta Llama 3](https://ai.meta.com/llama/)
- [Ramsey Theory](https://en.wikipedia.org/wiki/Ramsey_theory)

## TODO List

The following tasks are available for contributors looking to get involved:

### For Newcomers
1. **Setup Environment**: Complete the installation steps and run the test examples to ensure everything works
2. **Code Walkthrough**: Read through `l_shape_ramsey.py` and `llama_funsearch.py` to understand core algorithms
3. **Run Small Tests**: Try solving for grid sizes 3-5 with different parameters
4. **Review Documentation**: Familiarize yourself with the project documentation and experiment results
5. **Explore Visualizations**: Run the visualization tools to understand how solutions are represented

### Current Development Tasks
1. **SAT Solver Integration**: Implement Boolean satisfiability solvers to find valid L-shape colorings
   
   #### Understanding the Approach
   The L-shape Ramsey problem can be encoded as a Boolean satisfiability (SAT) problem by creating variables that represent the color assignments for each cell and constraints that ensure no monochromatic L-shapes exist.
   
   #### Implementation Plan
   - **Variable Encoding**: Create Boolean variables x_{i,j,c} where:
     - i, j are the grid coordinates (0 to n-1)
     - c is the color (0, 1, 2)
     - x_{i,j,c} = True means cell (i,j) has color c
   
   - **Core Constraints**:
     1. **Color Assignment**: Every cell must have exactly one color
        - For each cell (i,j): (x_{i,j,0} ∨ x_{i,j,1} ∨ x_{i,j,2})
        - For each cell (i,j) and colors c₁≠c₂: (¬x_{i,j,c₁} ∨ ¬x_{i,j,c₂})
     
     2. **No Monochromatic L-shapes**: For each L-shape pattern and color c:
        - ¬(x_{i,j,c} ∧ x_{i+d,j,c} ∧ x_{i+d,j+d,c}) for "Right and Up" L-shapes
        - ¬(x_{i,j,c} ∧ x_{i+d,j,c} ∧ x_{i+d,j-d,c}) for "Right and Down" L-shapes
        - ¬(x_{i,j,c} ∧ x_{i-d,j,c} ∧ x_{i-d,j+d,c}) for "Left and Up" L-shapes
        - ¬(x_{i,j,c} ∧ x_{i-d,j,c} ∧ x_{i-d,j-d,c}) for "Left and Down" L-shapes
        - These can be rewritten in CNF as: (¬x_{i,j,c} ∨ ¬x_{i+d,j,c} ∨ ¬x_{i+d,j+d,c})
   
   #### Recommended SAT Solvers
   1. **PySAT**: Python toolkit for SAT-based prototyping
      - Advantage: Clean Python API with multiple solver backends
      - Integration: `pip install python-sat`
   
   2. **Z3**: Microsoft's powerful SMT solver with Python bindings
      - Advantage: Can express higher-level constraints and handle SMT problems
      - Integration: `pip install z3-solver`
   
   #### Suggested Implementation Steps
   1. Create a module `sat_l_shape_solver.py` with a `SatLShapeSolver` class
   2. Implement constraint generation functions for L-shape patterns
   3. Add a CNF formatter that converts L-shape constraints to clauses
   4. Wrap solver calls to PySAT/Z3 and interpret results back to grid assignments
   5. Create benchmarking tools to compare against other approaches
   
   
   #### Useful Resources
   - [PySAT Documentation](https://pysathq.github.io/) - Complete Python SAT solving toolkit
   - [Z3 Tutorial](https://github.com/Z3Prover/z3/blob/master/examples/python/tutorial.py) - Introduction to using Z3 with Python
   - [SAT for Graph Coloring](https://arxiv.org/abs/1107.4375) - Paper on encoding coloring problems as SAT
   - [Handbook of Satisfiability](https://www.iospress.com/catalog/books/handbook-of-satisfiability-2) - Comprehensive reference on SAT solving techniques
   - [SAT Competition](https://satcompetition.github.io/) - Repository of advanced SAT solver benchmarks