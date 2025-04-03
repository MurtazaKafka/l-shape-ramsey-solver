#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, List, Tuple
import time
import torch
import argparse
import sys
from pathlib import Path
from llama import Llama

class Llama3LShapeSolver:
    """L-shape Ramsey problem solver using Llama 3.2."""
    
    def __init__(self, model_path=None):
        """Initialize the solver with the Llama 3.2 model."""
        if model_path is None:
            # Default to the Llama 3.2 model location
            model_path = Path.home() / ".llama" / "checkpoints" / "Llama3.2-3B-Instruct"
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please check the path.")
        
        print(f"Loading Llama 3.2 model from {self.model_path}")
        self.model = Llama(
            model_path=str(self.model_path),
            n_ctx=2048,  # Context length
            n_gpu_layers=-1  # Use all available GPU layers
        )
        
        # Cache for solutions
        self.solutions = {}
        
        # System prompt for the L-shape Ramsey problem
        self.system_prompt = """You are an expert in combinatorial optimization and Ramsey Theory.
        
Task: Solve the L-shape Ramsey problem for an N×N grid.
Context: The L-shape Ramsey problem asks for a 2-coloring of an N×N grid such that no L-shape is monochromatic.
Definition: An L-shape consists of 3 cells in a line (horizontally or vertically) plus 1 cell adjacent to the middle cell, forming an L.

Return a function that implements a solution pattern. Your function should:
1. Take grid_size as input
2. Return a 2D numpy array with values 0 or 1 representing the two colors
3. Ensure no L-shapes of the same color exist

Focus on mathematical patterns like modular arithmetic or recursive structures that provably avoid L-shapes.
"""
    
    def verify_solution(self, grid: np.ndarray) -> bool:
        """Verify if a given grid coloring is valid (no monochromatic L-shapes)."""
        n = len(grid)
        
        # Check all possible L-shapes
        for i in range(n-2):
            for j in range(n-2):  # Changed to n-2 to avoid index out of bounds
                # Check horizontal L-shapes
                if grid[i,j] == grid[i,j+1] == grid[i,j+2] == grid[i+1,j+1]:
                    print(f"Found invalid horizontal L-shape at ({i},{j})")
                    return False
                    
                # Check vertical L-shapes
                if grid[i,j] == grid[i+1,j] == grid[i+2,j] == grid[i+1,j+1]:
                    print(f"Found invalid vertical L-shape at ({i},{j})")
                    return False
                    
                if j > 0:
                    # Check rotated L-shapes
                    if grid[i,j] == grid[i+1,j] == grid[i+2,j] == grid[i+1,j-1]:
                        print(f"Found invalid rotated L-shape at ({i},{j})")
                        return False
                        
        # Check rotated horizontal L-shapes
        for i in range(1, n):
            for j in range(n-2):
                if grid[i,j] == grid[i,j+1] == grid[i,j+2] == grid[i-1,j+1]:
                    print(f"Found invalid rotated horizontal L-shape at ({i},{j})")
                    return False
                    
        return True
    
    def generate_solution(self, grid_size: int) -> Optional[np.ndarray]:
        """Generate a solution function using Llama 3.2."""
        prompt = f"""For a {grid_size}×{grid_size} grid, provide a function that creates a valid coloring for the L-shape Ramsey problem.

An example solution for a 3×3 grid uses the pattern:
```python
def generate_grid(n):
    grid = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            grid[i, j] = (i + 2*j) % 3 % 2
    return grid
```

Focus on efficient mathematical patterns that work for size {grid_size}.
"""
        
        # Generate a response from the model
        response = self.model.generate(
            prompt=[prompt],
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            stop=["<end>"]
        )
        
        generated_text = response[0]
        print("Generated solution:\n", generated_text)
        
        # Extract the Python function from the generated text
        try:
            # Find the function definition
            start_idx = generated_text.find("def ")
            if start_idx == -1:
                print("No function definition found")
                return None
                
            end_idx = generated_text.find("```", start_idx)
            if end_idx == -1:
                end_idx = len(generated_text)
                
            function_text = generated_text[start_idx:end_idx].strip()
            
            # Create a temporary file to store the function
            with open("temp_solution.py", "w") as f:
                f.write("import numpy as np\n\n")
                f.write(function_text)
                f.write("\n\nif __name__ == '__main__':\n")
                f.write(f"    grid = generate_grid({grid_size})\n")
                f.write("    print(grid)")
            
            # Import and execute the function
            import sys
            import importlib.util
            
            # Load the module
            spec = importlib.util.spec_from_file_location("temp_solution", "temp_solution.py")
            temp_solution = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_solution)
            
            # Call the function
            solution = temp_solution.generate_grid(grid_size)
            
            # Clean up
            os.remove("temp_solution.py")
            
            return solution
            
        except Exception as e:
            print(f"Error executing generated solution: {e}")
            return None
    
    def solve(self, grid_size: int, max_attempts: int = 3) -> Optional[np.ndarray]:
        """Solve the L-shape Ramsey problem for the given grid size."""
        
        # Check if solution is already cached
        if grid_size in self.solutions:
            print(f"Using cached solution for {grid_size}×{grid_size}")
            return self.solutions[grid_size]
        
        print(f"Solving {grid_size}×{grid_size} grid...")
        
        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1}/{max_attempts}")
            solution = self.generate_solution(grid_size)
            
            if solution is not None and self.verify_solution(solution):
                print(f"Found valid solution for {grid_size}×{grid_size} grid!")
                self.solutions[grid_size] = solution
                self.visualize_solution(solution, grid_size, "Llama 3.2")
                return solution
        
        print(f"No valid solution found for {grid_size}×{grid_size} grid after {max_attempts} attempts")
        return None
    
    def visualize_solution(self, grid: np.ndarray, grid_size: int, method: str = ""):
        """Visualize the solution grid."""
        plt.figure(figsize=(8, 8))
        cmap = plt.cm.colors.ListedColormap(['red', 'blue'])
        plt.imshow(grid, cmap=cmap)
        plt.grid(True, color='black', linewidth=1.5)
        plt.xticks(range(grid_size))
        plt.yticks(range(grid_size))
        title = f"L-shape Ramsey {grid_size}×{grid_size} Grid Solution"
        if method:
            title += f" ({method})"
        plt.title(title)
        
        # Save the visualization
        os.makedirs("visualizations", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"visualizations/l_shape_grid_{grid_size}x{grid_size}_{timestamp}.png")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Solve L-shape Ramsey problem using Llama 3.2')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the Llama 3.2 model (default: ~/.llama/checkpoints/Llama3.2-3B-Instruct)')
    parser.add_argument('--grid-sizes', type=int, nargs='+', default=[3, 4, 5, 6, 7],
                        help='Grid sizes to solve (default: 3 4 5 6 7)')
    parser.add_argument('--max-attempts', type=int, default=3,
                        help='Maximum attempts per grid size (default: 3)')
    args = parser.parse_args()
    
    try:
        # Initialize solver
        solver = Llama3LShapeSolver(args.model_path)
        
        # Test on different grid sizes
        grid_sizes = args.grid_sizes
        
        results = {}
        
        for size in grid_sizes:
            solution = solver.solve(size, args.max_attempts)
            
            if solution is not None:
                print(f"Found valid solution for {size}×{size} grid:")
                print(solution)
                results[size] = "Solved"
            else:
                print(f"No valid solution found for {size}×{size} grid")
                results[size] = "Unsolved"
        
        print("\nSummary of results:")
        for size, result in results.items():
            print(f"{size}×{size} grid: {result}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 