#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, List, Tuple
import time
import argparse
import sys
from pathlib import Path
import tempfile
import importlib.util
import json
import requests

class OllamaSolver:
    """L-shape Ramsey problem solver using Ollama with Llama 3.2."""
    
    def __init__(self, model_name="llama3.2:3b", max_tokens=1024, temperature=0.7):
        """Initialize the solver with the Ollama model."""
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_url = "http://localhost:11434/api/generate"  # Use generate endpoint instead of chat
        
        print(f"Using Ollama with model: {model_name}")
        
        # Test connection to Ollama
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                if self.model_name in model_names:
                    print(f"Found {self.model_name} in available models")
                else:
                    print(f"Warning: {self.model_name} not found in available models: {model_names}")
            else:
                print(f"Warning: Failed to get available models. Status code: {response.status_code}")
        except Exception as e:
            print(f"Warning: Failed to connect to Ollama API: {e}")
            print("Make sure Ollama is running with 'brew services start ollama'")
        
        # Cache for solutions
        self.solutions = {}
    
    def verify_solution(self, grid: np.ndarray) -> bool:
        """Verify if a given grid coloring is valid (no monochromatic L-shapes)."""
        n = len(grid)
        
        # Check all possible L-shapes as defined in the original solver
        for i in range(n):
            for j in range(n):
                color = grid[i, j]
                
                # Check all possible L-shapes starting from this point
                for d in range(1, n):
                    # Check all four orientations of L-shapes
                    
                    # Right and Up
                    if (j + d) < n and (i + d) < n:
                        if (grid[i, j+d] == color and 
                            grid[i+d, j+d] == color):
                            print(f"Found invalid L-shape (Right+Up) at ({i},{j}) with d={d}")
                            return False
                    
                    # Right and Down
                    if (j + d) < n and (i - d) >= 0:
                        if (grid[i, j+d] == color and 
                            grid[i-d, j+d] == color):
                            print(f"Found invalid L-shape (Right+Down) at ({i},{j}) with d={d}")
                            return False
                    
                    # Left and Up
                    if (j - d) >= 0 and (i + d) < n:
                        if (grid[i, j-d] == color and 
                            grid[i+d, j-d] == color):
                            print(f"Found invalid L-shape (Left+Up) at ({i},{j}) with d={d}")
                            return False
                    
                    # Left and Down
                    if (j - d) >= 0 and (i - d) >= 0:
                        if (grid[i, j-d] == color and 
                            grid[i-d, j-d] == color):
                            print(f"Found invalid L-shape (Left+Down) at ({i},{j}) with d={d}")
                            return False
        
        return True
    
    def generate_solution(self, grid_size: int) -> Optional[np.ndarray]:
        """Generate a solution function using Ollama."""
        system_prompt = """You are an expert in combinatorial optimization and Ramsey Theory.
        
Task: Solve the L-shape Ramsey problem for an N×N grid using THREE colors.
Context: The L-shape Ramsey problem asks for a 3-coloring of an N×N grid such that no L-shape is monochromatic.
Definition: An L-shape consists of three points where two points are equidistant from the third point, forming a right angle.

For example, these are L-shapes:
- Points at (0,0), (2,0), and (2,2) form an L-shape
- Points at (1,1), (1,3), and (3,3) form an L-shape
- Points at (4,2), (2,2), and (2,0) form an L-shape

Your goal is to create a Python function that returns a valid 3-coloring for the grid.
"""

        user_prompt = f"""For a {grid_size}×{grid_size} grid, provide a function that creates a valid 3-coloring for the L-shape Ramsey problem.

For reference, here are example solutions for smaller grids:

For a 3×3 grid:
```python
def generate_grid(n):
    grid = np.zeros((n, n), dtype=int)
    # Latin square pattern is known to work well
    pattern = [
        [0, 1, 2],
        [2, 0, 1],
        [1, 2, 0]
    ]
    for i in range(n):
        for j in range(n):
            grid[i, j] = pattern[i % 3][j % 3]
    return grid
```

For a 4×4 grid:
```python
def generate_grid(n):
    grid = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            grid[i, j] = (i + 2*j) % 3
    return grid
```

Return ONLY a Python function named 'generate_grid' that works for size {grid_size}. 
The function should:
1. Take grid_size as input parameter 'n'
2. Return a numpy array with values 0, 1, or 2 (representing red, green, blue)
3. Use mathematical patterns to ensure no L-shapes are monochromatic
"""

        # Generate text with Ollama
        try:
            # Use the simpler generate endpoint instead of chat
            prompt = f"{system_prompt}\n\n{user_prompt}"
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False  # Don't stream the response
            }
            
            print(f"Sending request to Ollama for {grid_size}×{grid_size} grid...")
            response = requests.post(self.api_url, json=payload)
            
            if response.status_code != 200:
                print(f"Error from Ollama API: {response.status_code} - {response.text}")
                return None
            
            # Extract the generated text
            try:
                response_json = response.json()
                generated_text = response_json.get("response", "")
                
                # Print a preview of the generated text
                if generated_text:
                    print(f"\nGenerated solution preview:\n{generated_text[:500]}...\n")
                else:
                    print("Warning: Empty response from Ollama")
                    return None
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print(f"Response content:\n{response.text[:200]}...")
                return None
            
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
                def_pos = function_code.find("def ")
                if def_pos != -1:
                    function_code = function_code[def_pos:]
                else:
                    print("Error: No function definition found")
                    return None
            
            print(f"Extracted function code:\n{function_code}\n")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(b"import numpy as np\n\n")
                temp_file.write(function_code.encode())
            
            try:
                # Import the module
                module_name = Path(temp_file_path).stem
                spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Call the function
                solution = module.generate_grid(grid_size)
                
                # Ensure solution is a numpy array with int elements
                if not isinstance(solution, np.ndarray):
                    solution = np.array(solution)
                
                if solution.dtype != int:
                    solution = solution.astype(int)
                
                return solution
            except Exception as e:
                print(f"Error executing function: {e}")
                # Print the file content for debugging
                with open(temp_file_path, 'r') as f:
                    print(f"File content:\n{f.read()}")
                return None
            finally:
                # Clean up
                os.unlink(temp_file_path)
            
        except Exception as e:
            print(f"Error generating solution: {e}")
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
            
            # Wait a bit before next attempt
            if attempt < max_attempts - 1:
                print("Waiting for 5 seconds before next attempt...")
                time.sleep(5)
        
        print(f"No valid solution found for {grid_size}×{grid_size} grid after {max_attempts} attempts")
        return None
    
    def visualize_solution(self, grid: np.ndarray, grid_size: int, method: str = ""):
        """Visualize the solution grid."""
        plt.figure(figsize=(8, 8))
        
        # Use three colors: red, green, blue
        if grid.max() <= 1:
            # Binary grid (0, 1)
            cmap = plt.cm.colors.ListedColormap(['red', 'blue'])
        else:
            # Three-color grid (0, 1, 2)
            cmap = plt.cm.colors.ListedColormap(['red', 'green', 'blue'])
            
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
        plt.savefig(f"visualizations/llama3_grid_{grid_size}x{grid_size}_{timestamp}.png")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Solve L-shape Ramsey problem using Ollama with Llama 3.2')
    parser.add_argument('--model-name', type=str, default="llama3.2:3b",
                        help='Model name for Ollama (default: llama3.2:3b)')
    parser.add_argument('--grid-sizes', type=int, nargs='+', default=[3, 4, 5],
                        help='Grid sizes to solve (default: 3 4 5)')
    parser.add_argument('--max-attempts', type=int, default=5,
                        help='Maximum attempts per grid size (default: 5)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for text generation (default: 0.7)')
    args = parser.parse_args()
    
    try:
        # Initialize solver
        solver = OllamaSolver(
            model_name=args.model_name,
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