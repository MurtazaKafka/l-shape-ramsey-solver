# L-Shape Ramsey Problem Solver

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Definition](#problem-definition)
3. [Solution Approaches](#solution-approaches)
4. [Project Structure](#project-structure)
5. [Core Components](#core-components)
6. [Running the Code](#running-the-code)
7. [Algorithm Details](#algorithm-details)
8. [Visualization Tools](#visualization-tools)
9. [Advanced Usage](#advanced-usage)
10. [Extending the Project](#extending-the-project)
11. [Results & Performance](#results--performance)
12. [Troubleshooting](#troubleshooting)
13. [References](#references)

## Project Overview

This project implements several approaches to solve the L-shape Ramsey problem, with a focus on using AI language models (particularly Meta's Llama 3.3 70B) to discover and optimize solutions through FunSearch. The implementations range from deterministic algorithms to reinforcement learning and large language model-based approaches.

The central goal is to find valid colorings of n×n grids using three colors such that no L-shape has all three points colored the same (monochromatic). While a solution is known for the 3×3 grid, finding valid colorings for larger grid sizes is challenging.

## Problem Definition

**The L-Shape Ramsey Problem:**
- We have an n×n grid with each cell colored using one of three colors (typically labeled 0, 1, and 2, or Red, Green, and Blue)
- An L-shape consists of three points: a corner point and two points equidistant from it forming a right angle
- A coloring is valid if no L-shape is monochromatic (has all three points colored the same)

Examples of L-shapes:
- Points at (0,0), (2,0), and (2,2) form an L-shape
- Points at (1,1), (1,3), and (3,3) form an L-shape
- Points at (4,2), (2,2), and (2,0) form an L-shape

The challenge is to find valid colorings for increasingly large grid sizes, as the search space grows exponentially.

## Solution Approaches

This project implements several approaches:

1. **Deterministic Patterns**: Mathematical patterns like Latin squares that create valid colorings
2. **FunSearch Algorithm**: Evolving code to generate valid colorings using language models
3. **Reinforcement Learning**: Training models to create valid colorings
4. **Hierarchical Search**: Breaking the grid into smaller blocks to solve independently
5. **GPU Acceleration**: Using GPU for parallel search of the solution space

## Project Structure

The project contains several key components:

- **Core Components**: `l_shape_ramsey.py` - defines the grid representation and validation
- **AI/ML Solutions**:
  - `llama_funsearch.py` - implementation of FunSearch with Llama 3.3 70B
  - `l_shape_rl.py` - reinforcement learning approach
- **Deterministic Solutions**:
  - `fixed_solver.py` - deterministic patterns
  - `specialized_4x4_solver.py` - specialized solver for 4×4 grids
- **Visualization Tools**: For rendering and analyzing grid colorings
- **Utility Scripts**: For GPU management, running experiments, etc.
- **Results Storage**: Directories containing solution grids and visualizations

## Core Components

### LShapeGrid Class (`l_shape_ramsey.py`)

The central class representing a colored grid:

```python
class LShapeGrid:
    def __init__(self, size: int):
        self.size = size
        self.grid = np.full((size, size), None, dtype=object)
    
    def set_color(self, x: int, y: int, color: Color) -> None:
        """Set the color at position (x,y)"""
        # ...existing code...
    
    def get_color(self, x: int, y: int) -> Optional[Color]:
        """Get the color at position (x,y)"""
        # ...existing code...
    
    def has_l_shape(self, x: int, y: int) -> Tuple[bool, List[Tuple[int, int]]]:
        """Check if there's a monochromatic L-shape starting at position (x,y)"""
        # ...existing code...
    
    def has_any_l_shape(self) -> Tuple[bool, List[Tuple[int, int]]]:
        """Check if the grid contains any monochromatic L-shape"""
        # ...existing code...
    
    def visualize(self, highlight_l_shape: bool = True, filename: str = None):
        """Visualize the grid"""
        # ...existing code...
```

### FunSearch Implementation (`llama_funsearch.py`)

The main implementation of the FunSearch algorithm with Meta's Llama 3.3 70B model:

```python
class LlamaFunSearch:
    def __init__(self, model_path=None, temperature=0.7, max_tokens=2048):
        """Initialize the FunSearch solver with a Llama model"""
        # ...existing code...
    
    def solve(self, grid_size, iterations=10, time_limit=300):
        """Main method to solve the problem for a given grid size"""
        # ...existing code...
    
    def _create_latin_square(self, n):
        """Create a Latin square pattern for the baseline"""
        # ...existing code...
    
    def _verify_grid(self, grid):
        """Verify if a grid has monochromatic L-shapes"""
        # ...existing code...
```

## Running the Code

### Installation

1. Clone the repository:
   ```bash
   git clone <https://github.com/MurtazaKafka/l-shape-ramsey-solver>
   cd l-shape-ramsey-solver
   ```

2. Install dependencies:
   ```bash
   ./install_requirements.sh
   ```

3. Set up the environment for Llama 3.3 70B (if using FunSearch):
   ```bash
   conda create -n ramsey python=3.10
   conda activate ramsey
   CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
   ```

### Basic Usage

#### Running the Deterministic Solver

```bash
python fixed_solver.py --grid-size 5
```

#### Running FunSearch with Llama

```bash
python llama_funsearch.py --grid-size 4 --max-grid-size 6 --iterations 5 --time-limit 300
```

#### Testing Grid Visualizations

```bash
python test_visualization.py
```

## Algorithm Details

### Deterministic Patterns

Several deterministic patterns have been found to work for different grid sizes:

1. **Latin Square Pattern** (works for 3×3 grids):
   ```
   0 1 2
   2 0 1
   1 2 0
   ```

2. **Modular Arithmetic Pattern** (generalized approach):
   ```python
   def create_modular_grid(size):
       grid = np.zeros((size, size), dtype=int)
       for i in range(size):
           for j in range(size):
               grid[i, j] = (i + j*2) % 3
       return grid
   ```

### FunSearch Algorithm

FunSearch works by:

1. Starting with a baseline solution (Latin square or another pattern)
2. Using a language model (Llama 3.3 70B) to generate Python functions that create grid colorings
3. Evaluating each function by checking if the resulting grid has any L-shapes
4. Selecting the best functions for the next generation
5. Repeating the process for multiple iterations

The implementation uses an "island model" where multiple populations evolve independently.

## Visualization Tools

The project includes several visualization tools:

1. **Base Visualization** (`LShapeGrid.visualize`): Renders a grid with optional highlighting of L-shapes

2. **Enhanced Visualization** (`_visualize_grid` in `llama_funsearch.py`): Provides better visualization for larger grids

3. **Comparative Visualization** (`visualize_multiple_grids`): Combines multiple grid sizes for comparison

Example visualization output:
- `grid_4x4_solution.png`: The best solution found for a 4×4 grid
- `test_visualizations/`: Directory containing test visualizations for different patterns

## Advanced Usage

### Working with Larger Grids

For larger grids (8×8 and above), memory and computation time become constraints. Use these options:

```bash
python llama_funsearch.py --grid-size 8 --iterations 2 --time-limit 600 --output-dir large_grid_results
```

### Customizing the Search

To modify search parameters for FunSearch:

```bash
python llama_funsearch.py --temperature 0.9 --grid-size 4 --iterations 10
```

### Using GPU Acceleration

The code can use GPU acceleration for both the Llama model and grid operations:

```bash
python run_gpu_model.py --model-path "models/Llama-3.3-70B-Instruct-Q4_K_M.gguf"
```

## Extending the Project

### Adding New Patterns

To add a new deterministic pattern:

1. Create a new function in `fixed_solver.py`:
   ```python
   def my_new_pattern(size):
       grid = np.zeros((size, size), dtype=int)
       # Implement your pattern
       return grid
   ```

2. Register it in the `get_available_patterns` function.

### Working with Different Models

The FunSearch implementation can work with different models:

1. GGUF Llama models (default)
2. Hugging Face Transformers models
3. Alternative models via adapter classes

To use a different model:
```bash
python llama_funsearch.py --model-path "path/to/your/model"
```

## Results & Performance

### Current Results

| Grid Size | Solution Status | Method |
|-----------|----------------|--------|
| 3×3       | ✓ Solved       | Latin Square |
| 4×4       | ✓ Solved       | Specialized 4×4 Solver |
| 5×5       | ✓ Solved       | FunSearch + Optimization |
| 6×6       | ✓ Solved       | FunSearch + Block Patterns |
| 7×7       | ✓ Solved       | FunSearch + Hierarchical |
| 8×8       | ✓ Solved       | FunSearch + Hierarchical |
| 9×9       | ✓ Solved       | FunSearch + Hierarchical |
| 10×10     | ✓ Solved       | FunSearch + Hierarchical |
| >10×10    | Partial solutions | Scaling approaches |

### Performance Considerations

- **Memory Usage**: The 70B Llama model requires ~40GB VRAM with quantization
- **Computation Time**: Scales exponentially with grid size
- **Optimization**: The code includes various optimizations for handling larger grids

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**:
   - Error: "CUDA out of memory"
   - Solution: Use the GPU manager to free memory: `python gpu_manager.py --kill-all --force`

2. **Model Loading Issues**:
   - Error: "Failed to load model"
   - Solution: Ensure the model is in the correct path or use a smaller model

3. **Visualization Problems**:
   - Issue: Grids appear blank or L-shapes not highlighted
   - Solution: Check matplotlib version and ensure directory permissions are correct

### Debugging

For detailed debugging output:
```bash
python llama_funsearch.py --grid-size 4 --max-grid-size 4 --iterations 1 --time-limit 60 --output-dir test_results
```

## References

1. Ramsey Theory - [Wikipedia](https://en.wikipedia.org/wiki/Ramsey_theory)
2. FunSearch: Making new discoveries in mathematical sciences using LLMs - [DeepMind blog](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)
3. Meta Llama 3 - [Meta AI](https://ai.meta.com/llama/)

---

## Additional Resources

### Understanding L-shapes

An L-shape consists of three points with specific geometric properties:
- A corner point (x,y)
- Two points that are equidistant from the corner and form a right angle

For example, in a 5×5 grid, these form L-shapes:
- (0,0), (0,2), (2,2) - corner at (0,0)
- (2,2), (4,2), (4,4) - corner at (4,4)

### Performance Metrics

When evaluating grid colorings, we use several metrics:
- **Validity**: Whether the grid has any monochromatic L-shapes
- **Score**: A numerical score based on color distribution and pattern quality
- **Generation Time**: How long it takes to generate a valid solution

Higher scores indicate better solutions with better color distribution.

### Documentation Changelog

- **April 2025**: Initial documentation created
- **March 2025**: Updated with latest results for larger grid sizes
- **February 2025**: Added details on hierarchical search approach# Updated: Wed Apr 30 16:11:49 EDT 2025
