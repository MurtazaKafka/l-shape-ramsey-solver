#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, List, Tuple, Dict, Any
import time
import argparse
import sys
from pathlib import Path
import tempfile
import importlib.util
import json

# Import the llama_stack package
try:
    from llama_stack import LlamaStackAsLibraryClient
    from llama_stack.apis.datatypes import Api
    from llama_stack.providers.datatypes import ProviderSpec
except ImportError:
    print("Error: Failed to import 'LlamaStackAsLibraryClient' from 'llama_stack' package.")
    print("Try installing it with: pip install llama_stack")
    sys.exit(1)

class LlamaStackSolver:
    """L-shape Ramsey problem solver using Llama 3.2 via llama_stack."""
    
    def __init__(self, model_id="Llama3.2-3B-Instruct", max_tokens=2048, temperature=0.7):
        """Initialize the solver with the Llama model."""
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        print(f"Initializing Llama client for model: {model_id}")
        
        # Create a temporary config file for LlamaStackAsLibraryClient
        config = {
            "llm_provider": "local_model",
            "llm_model": model_id,
            "default_model_kwargs": {
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        }
        
        # Create a temporary config file
        config_path = "llama_stack_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        try:
            # Create LlamaStackAsLibraryClient with the config path
            self.client = LlamaStackAsLibraryClient(config_path_or_template_name=config_path)
            print(f"Successfully initialized client")
        except Exception as e:
            print(f"Error initializing Llama client: {e}")
            # Clean up the config file
            if os.path.exists(config_path):
                os.remove(config_path)
            raise
        
        # Clean up the config file
        if os.path.exists(config_path):
            os.remove(config_path)
        
        # Cache for solutions
        self.solutions = {}
    
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
        """Generate a solution function using Llama."""
        system_prompt = """You are an expert in combinatorial optimization and Ramsey Theory.
        
Task: Solve the L-shape Ramsey problem for an N×N grid.
Context: The L-shape Ramsey problem asks for a 2-coloring of an N×N grid such that no L-shape is monochromatic.
Definition: An L-shape consists of 3 cells in a line (horizontally or vertically) plus 1 cell adjacent to the middle cell, forming an L.

Your goal is to create a Python function that returns a valid coloring for the grid.
"""

        user_prompt = f"""For a {grid_size}×{grid_size} grid, provide a function that creates a valid coloring for the L-shape Ramsey problem.

An example solution for a 3×3 grid uses the pattern:
```python
def generate_grid(n):
    grid = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            grid[i, j] = (i + 2*j) % 3 % 2
    return grid
```

Return ONLY a Python function named 'generate_grid' that works for size {grid_size}. 
The function should:
1. Take grid_size as input parameter 'n'
2. Return a numpy array with values 0 or 1
3. Use mathematical patterns to ensure no L-shapes are monochromatic
"""

        # Generate text with Llama
        try:
            # Prepare messages for chat completion
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call the Llama client for chat completion
            response = self.client.chat_completion(
                model=self.model_id,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Extract the generated text
            generated_text = response["choices"][0]["message"]["content"]
            print(f"\nGenerated solution:\n{generated_text}\n")
            
            # Extract Python code
            code_start = generated_text.find("```python")
            if code_start == -1:
                code_start = generated_text.find("```")
                if code_start == -1:
                    code_start = generated_text.find("def generate_grid")
                else:
                    code_start += 3  # Skip ```
            else:
                code_start += 8  # Skip ```python
            
            code_end = generated_text.find("```", code_start)
            if code_end == -1:
                code_end = len(generated_text)
            
            # Extract the function code
            function_code = generated_text[code_start:code_end].strip()
            
            # Make sure the code starts with def
            if not function_code.startswith("def"):
                function_code = function_code[function_code.find("def"):]
            
            print(f"Extracted function code:\n{function_code}\n")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(b"import numpy as np\n\n")
                temp_file.write(function_code.encode())
            
            # Import the module
            module_name = Path(temp_file_path).stem
            spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Clean up
            os.unlink(temp_file_path)
            
            # Call the function
            solution = module.generate_grid(grid_size)
            
            # Ensure solution is a numpy array with int elements
            if not isinstance(solution, np.ndarray):
                solution = np.array(solution)
            
            if solution.dtype != int:
                solution = solution.astype(int)
            
            return solution
            
        except Exception as e:
            print(f"Error generating or executing solution: {e}")
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
            
            if solution is not None:
                print(f"Generated solution shape: {solution.shape}")
                
                # Ensure the solution is the right shape
                if solution.shape != (grid_size, grid_size):
                    print(f"Warning: Solution has incorrect shape {solution.shape}, expected ({grid_size}, {grid_size})")
                    continue
                
                # Verify the solution
                if self.verify_solution(solution):
                    print(f"Found valid solution for {grid_size}×{grid_size} grid!")
                    self.solutions[grid_size] = solution
                    self.visualize_solution(solution, grid_size, "Llama3")
                    return solution
                else:
                    print(f"Solution verification failed")
        
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
    parser.add_argument('--model-id', type=str, default="Llama3.2-3B-Instruct",
                        help='Model ID for Llama (default: Llama3.2-3B-Instruct)')
    parser.add_argument('--grid-sizes', type=int, nargs='+', default=[3, 4, 5, 6, 7],
                        help='Grid sizes to solve (default: 3 4 5 6 7)')
    parser.add_argument('--max-attempts', type=int, default=3,
                        help='Maximum attempts per grid size (default: 3)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for text generation (default: 0.7)')
    args = parser.parse_args()
    
    try:
        # Initialize solver
        solver = LlamaStackSolver(
            model_id=args.model_id,
            temperature=args.temperature
        )
        
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