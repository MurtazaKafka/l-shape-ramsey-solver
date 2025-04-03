#!/usr/bin/env python3
import os
import time
import json
import random
import argparse
import requests
import numpy as np
from datetime import datetime
from typing import Optional
import matplotlib.pyplot as plt
import tempfile
import importlib.util

from l_shape_ramsey import LShapeGrid, Color

class LlamaFunSearch:
    """
    A simplified FunSearch implementation for the L-shape Ramsey problem using Llama via Ollama.
    This implementation focuses on the 3×3 grid size with the verified Latin square pattern.
    """
    
    def __init__(self, model_name="llama3.2:3b", temperature=0.7, max_tokens=2048):
        """Initialize the LlamaFunSearch."""
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = "http://localhost:11434/api/generate"
        self.function_name = "generate_grid"
        
        # Store best solution
        self.best_score = 0
        self.best_solution = None
        
        # Starting pattern - Latin square (verified to work for 3×3)
        self.baseline_pattern = self._create_latin_square(3)
        
        # Output directory for results
        self.output_dir = "funsearch_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Test Ollama connection
        self._test_ollama_connection()
    
    def _test_ollama_connection(self):
        """Test the connection to Ollama."""
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
                print(f"Warning: Failed to get models from Ollama API: {response.status_code}")
        except Exception as e:
            print(f"Warning: Failed to connect to Ollama: {e}")
    
    def _create_latin_square(self, n):
        """Create a Latin square pattern for 3×3 grid."""
        grid = np.zeros((n, n), dtype=int)
        pattern = [
            [0, 1, 2],
            [2, 0, 1],
            [1, 2, 0]
        ]
        for i in range(min(n, 3)):
            for j in range(min(n, 3)):
                grid[i, j] = pattern[i][j]
        return grid
    
    def _verify_grid(self, grid):
        """Verify that a grid doesn't contain monochromatic L-shapes."""
        n = grid.shape[0]
        
        # Convert to LShapeGrid
        l_shape_grid = LShapeGrid(n)
        for i in range(n):
            for j in range(n):
                l_shape_grid.set_color(j, i, list(Color)[grid[i, j]])
        
        # Check for L-shapes
        has_l, points = l_shape_grid.has_any_l_shape()
        
        if has_l:
            return 0.0, points
        
        # If valid, calculate score
        score = 1.0
        
        # Favor diversity in rows and columns
        for i in range(n):
            score += len(set(grid[i, :])) / 3.0
            score += len(set(grid[:, i])) / 3.0
        
        return score, None
    
    def _generate_code(self, prompt):
        """Generate code using Llama model via Ollama."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload)
            
            if response.status_code != 200:
                print(f"Error from Ollama API: {response.status_code}")
                return ""
            
            response_json = response.json()
            return response_json.get("response", "")
        except Exception as e:
            print(f"Error generating code: {e}")
            return ""
    
    def _extract_code(self, text):
        """Extract code from the generated text."""
        # Try to find code blocks
        if "```python" in text:
            code_block = text.split("```python")[1].split("```")[0].strip()
            if self.function_name in code_block:
                return code_block
        
        elif "```" in text:
            code_block = text.split("```")[1].strip()
            if self.function_name in code_block:
                return code_block
        
        # Look for function definition directly
        if f"def {self.function_name}" in text:
            lines = text.split("\n")
            start_idx = -1
            for i, line in enumerate(lines):
                if f"def {self.function_name}" in line:
                    start_idx = i
                    break
            
            if start_idx >= 0:
                code_lines = []
                i = start_idx
                while i < len(lines):
                    code_lines.append(lines[i])
                    if i > start_idx and not (lines[i].startswith(" ") or lines[i].startswith("\t") or lines[i] == ""):
                        break
                    i += 1
                
                code = "\n".join(code_lines)
                return code
        
        return ""
    
    def _evaluate_function(self, code):
        """Evaluate a generated function."""
        try:
            # Create a module to execute the code
            with tempfile.NamedTemporaryFile(suffix='.py') as temp_file:
                # Add numpy import to avoid common import error
                full_code = "import numpy as np\n\n" + code
                temp_file.write(full_code.encode())
                temp_file.flush()
                
                spec = importlib.util.spec_from_file_location("temp_module", temp_file.name)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Check if the function exists
                if not hasattr(module, self.function_name):
                    return 0.0, None, "Function not found in generated code"
                
                # Get the function and execute it
                func = getattr(module, self.function_name)
                grid = func(3)  # We're focusing on 3×3 grid
                
                if not isinstance(grid, np.ndarray) or grid.shape != (3, 3):
                    return 0.0, None, "Function didn't return a valid 3×3 grid"
                
                # Verify no L-shapes and calculate score
                score, points = self._verify_grid(grid)
                
                if score > 0:
                    return score, grid, "Valid solution"
                else:
                    return 0.0, None, f"Invalid solution: L-shape at {points}"
                
        except Exception as e:
            return 0.0, None, f"Error evaluating function: {e}"
    
    def _create_prompt(self, iterations_run=0):
        """Create a prompt for the Llama model."""
        system_prompt = """You are an expert in combinatorial optimization and Ramsey Theory.

Task: Solve the L-shape Ramsey problem for a 3×3 grid using THREE colors (0, 1, 2).
Context: The L-shape Ramsey problem asks for a 3-coloring of a grid such that no L-shape is monochromatic.
Definition: An L-shape consists of three points where two points are equidistant from the third point, forming a right angle.

For example, these are L-shapes:
- Points at (0,0), (2,0), and (2,2) form an L-shape
- Points at (1,1), (1,3), and (3,3) form an L-shape
- Points at (4,2), (2,2), and (2,0) form an L-shape

Your goal is to create a Python function that returns a valid 3-coloring for the grid.
"""

        # Include the baseline pattern if we have one
        best_solution_text = ""
        if self.best_solution is not None:
            best_solution_text = f"""
For reference, here is a known valid solution for a 3×3 grid:
```
{self.best_solution}
```

This Latin square pattern is known to work. You can use it as a starting point or create an entirely new solution.
"""

        user_prompt = f"""Create a function named '{self.function_name}' that generates a valid 3-coloring for a 3×3 grid for the L-shape Ramsey problem.

The function should:
1. Take grid_size as input parameter 'n' (n=3)
2. Return a numpy array with values 0, 1, 2 (representing three colors)
3. Ensure no L-shapes are monochromatic

{best_solution_text}

Iteration #{iterations_run+1}: Try to find a valid or improved solution.

Helpful hints:
1. Latin square patterns (where each row and column contains each color once) work well
2. Modular arithmetic patterns like (i + j) % 3 can also work
3. Consider patterns where adjacent cells have different colors

Return ONLY the Python function without any explanation.
"""

        return f"{system_prompt}\n\n{user_prompt}"
    
    def _save_result(self, grid, score, code):
        """Save the result to disk."""
        # Create grid directory
        os.makedirs(os.path.join(self.output_dir, "grid_3"), exist_ok=True)
        
        # Save code
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, "grid_3", f"solution_3_{timestamp}.py")
        
        with open(filename, "w") as f:
            f.write(f"# Solution for 3x3 grid\n")
            f.write(f"# Score: {score}\n")
            f.write(f"# Generated: {timestamp}\n\n")
            f.write("import numpy as np\n\n")
            f.write(code)
        
        # Save visualization
        vis_filename = os.path.join(self.output_dir, "grid_3", f"grid_3_{timestamp}.png")
        self._visualize_grid(grid, vis_filename)
        
        print(f"Saved result to {filename}")
        print(f"Saved visualization to {vis_filename}")
    
    def _visualize_grid(self, grid, filename=None):
        """Visualize a grid."""
        plt.figure(figsize=(6, 6))
        cmap = plt.cm.colors.ListedColormap(['red', 'green', 'blue'])
        plt.imshow(grid, cmap=cmap, vmin=0, vmax=2)
        plt.grid(True, color='black', linewidth=1.5)
        plt.xticks(range(3))
        plt.yticks(range(3))
        plt.title("L-shape Ramsey 3×3 Grid Solution")
        
        if filename:
            plt.savefig(filename, dpi=150)
            plt.close()
        else:
            plt.show()
    
    def run(self, iterations=10, time_limit=300):
        """Run the FunSearch algorithm."""
        print("Starting FunSearch for 3×3 grid...")
        
        # Set our baseline solution
        baseline_score, _ = self._verify_grid(self.baseline_pattern)
        self.best_score = baseline_score
        self.best_solution = self.baseline_pattern
        
        print(f"Baseline Latin square solution with score {baseline_score}:")
        print(self.baseline_pattern)
        
        # Main loop
        start_time = time.time()
        iterations_run = 0
        
        while iterations_run < iterations and (time.time() - start_time) < time_limit:
            print(f"\nIteration {iterations_run+1}/{iterations}...")
            
            # Generate code
            prompt = self._create_prompt(iterations_run)
            generated_text = self._generate_code(prompt)
            
            # Extract code
            code = self._extract_code(generated_text)
            
            if code:
                # Evaluate function
                score, grid, message = self._evaluate_function(code)
                print(f"Evaluation: {message}")
                
                if score > 0:
                    print(f"Valid solution found! Score: {score}")
                    print(grid)
                    
                    # Update best solution if better
                    if score > self.best_score:
                        print(f"New best solution! Score: {score}")
                        self.best_score = score
                        self.best_solution = grid
                        self._save_result(grid, score, code)
            else:
                print("Failed to extract code from generation")
            
            iterations_run += 1
        
        # Final results
        elapsed_time = time.time() - start_time
        print(f"\nFunSearch completed in {elapsed_time:.1f} seconds")
        print(f"Best solution found (score: {self.best_score}):")
        print(self.best_solution)
        
        return self.best_solution, self.best_score

def main():
    parser = argparse.ArgumentParser(description='FunSearch for L-shape Ramsey problem (3×3 grid)')
    parser.add_argument('--iterations', type=int, default=10,
                      help='Maximum iterations (default: 10)')
    parser.add_argument('--time-limit', type=int, default=300,
                      help='Time limit in seconds (default: 300)')
    parser.add_argument('--model', type=str, default="llama3.2:3b",
                      help='Ollama model to use (default: llama3.2:3b)')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for generation (default: 0.7)')
    args = parser.parse_args()
    
    # Create and run FunSearch
    funsearch = LlamaFunSearch(
        model_name=args.model, 
        temperature=args.temperature
    )
    
    funsearch.run(args.iterations, args.time_limit)

if __name__ == "__main__":
    main() 