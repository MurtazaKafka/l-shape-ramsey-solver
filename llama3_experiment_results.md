# Llama 3.2 Experiment Results for L-shape Ramsey Problem

## Overview

We successfully integrated Meta's Llama 3.2 3B model with our L-shape Ramsey problem solver using Ollama as the inference engine. This document summarizes the results and compares the Llama-generated solutions with our deterministic approach.

## Summary of Results

| Grid Size | Deterministic Solver | Llama 3.2 | Pattern Used by Llama 3.2 |
|-----------|----------------------|-----------|---------------------------|
| 3×3       | ✅ Solved | ✅ Solved | `(i + 2*j) % 3` |
| 4×4       | ✅ Solved | ✅ Solved | `(i + 2*j) % 3` |
| 5×5       | ✅ Solved | ✅ Solved | `(i + 2*j) % 3 % 2` |
| 6×6       | ✅ Solved | ✅ Solved | `(i + 2*j) % 3 % 2` |
| 7×7       | ✅ Solved | ✅ Solved | `(i + 2*j) % 3 % 2` |
| 8×8       | ✅ Solved | ✅ Solved | `((i + 2*(j+1)) % 3) % 2` (modified pattern) |

## Analysis of Llama 3.2 Generated Solutions

### Key Observations

1. **Pattern Identification**: Llama 3.2 successfully identified effective mathematical patterns for coloring the grids.

2. **Mathematical Pattern**: The model consistently used modular arithmetic based on the formula `(i + 2*j) % 3 % 2` for most grid sizes, which aligns with the formula used in our deterministic solver.

3. **Pattern Adaptation**: For the 8×8 grid, Llama 3.2 modified the pattern slightly to `((i + 2*(j+1)) % 3) % 2` with an offset to ensure no monochromatic L-shapes formed.

4. **Learning from Examples**: The model effectively learned from the provided example and generalized the pattern to larger grid sizes.

5. **Algorithmic Thinking**: Llama 3.2 demonstrated algorithmic thinking by applying the same pattern with minor variations to successfully solve all grid sizes.

### Comparison with Deterministic Approach

Both approaches used essentially the same mathematical pattern for generating valid colorings:

```python
# Deterministic approach:
grid[i, j] = (i + 2*j) % 3 % 2

# Llama 3.2 approach (for 8×8):
grid[i, j] = ((i + 2*(j+1)) % 3) % 2  # Notice the offset (j+1)
```

The key findings from this comparison:

1. **Same Core Pattern**: Both approaches used modular arithmetic with a 3-coloring reduced to a 2-coloring.

2. **Efficiency**: Both solutions have O(n²) time complexity, iterating through each cell once.

3. **Adaptability**: Llama 3.2 showed some adaptability by slightly modifying the pattern for the 8×8 grid.

4. **Verification Success**: All solutions passed the verification step, confirming that no monochromatic L-shapes were present.

## Visualizations

Visualizations were created for all Llama 3.2 solutions and are stored in the `visualizations/` directory with the prefix `llama3_grid_`.

## Conclusion

Llama 3.2 successfully generated valid mathematical patterns for the L-shape Ramsey problem across all tested grid sizes (3×3 to 8×8). The model demonstrated the ability to:

1. Learn from examples
2. Apply mathematical patterns
3. Adapt patterns for different grid sizes
4. Generate code that produces valid solutions

This experiment shows that Llama 3.2 can be effectively used for combinatorial optimization problems where clear mathematical patterns exist. The model's ability to understand and generate valid solutions for the L-shape Ramsey problem demonstrates its potential for assisting with algorithmic problem-solving tasks.

Future work could explore more complex Ramsey-type problems where the solution patterns are less obvious or require more sophisticated mathematical reasoning. 