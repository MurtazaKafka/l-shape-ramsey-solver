#!/usr/bin/env python3
import os
import time
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple, Dict, Set, Optional
from pathlib import Path
import tempfile
import importlib.util
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import glob

from l_shape_ramsey import LShapeGrid, Color

class LlamaFunSearch:
    """
    A FunSearch implementation for the L-shape Ramsey problem using Llama via Transformers.
    This implementation focuses on the 3×3 grid size with the verified Latin square pattern.
    """
    
    def __init__(self, model_path=None, temperature=0.7, max_tokens=2048):
        """Initialize the LlamaFunSearch."""
        self.model_path = model_path or "/home/DAVIDSON/murtaza/.llama/checkpoints/Llama3.3-70B-Instruct"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.function_name = "generate_grid"
        
        # Store best solution
        self.best_score = 0
        self.best_solution = None
        
        # Starting pattern - Latin square (verified to work for 3×3)
        self.baseline_pattern = self._create_latin_square(3)
        
        # Output directory for results
        self.output_dir = "funsearch_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize retry parameters
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        
        # Initialize thread pool for parallel evaluation
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Load model and tokenizer
        self._load_model()
    
    def _find_model_path(self):
        """Find the model path if the specified one doesn't exist."""
        if os.path.exists(self.model_path):
            return self.model_path
            
        print(f"Model path {self.model_path} not found. Searching for alternative paths...")
        
        # Check parent directory first
        parent_dir = os.path.dirname(self.model_path)
        if os.path.exists(parent_dir):
            print(f"Using parent directory: {parent_dir}")
            return parent_dir
            
        # Check common locations
        common_locations = [
            # Home directories
            os.path.expanduser("~/.cache/huggingface/hub"),
            os.path.expanduser("~/models"),
            os.path.expanduser("~/llama"),
            
            # Shared directories
            "/shared/models",
            "/data/models",
            "/mnt/models",
            "/models",
            
            # Davidson-specific directories
            "/home/DAVIDSON/shared/models",
            "/home/models",
        ]
        
        # Look for meta-llama, llama3, llama-3, etc.
        for location in common_locations:
            if not os.path.exists(location):
                continue
                
            # Check for potential model directories
            for pattern in ["*llama*", "*Llama*", "*LLAMA*", "*70B*"]:
                matches = glob.glob(f"{location}/{pattern}")
                if matches:
                    print(f"Found potential model location: {matches[0]}")
                    return matches[0]
        
        # If all else fails, return the original path (will likely fail later)
        print("No alternative model paths found.")
        return self.model_path
    
    def _load_model(self):
        """Load the Llama model and tokenizer using Transformers."""
        # First try to find the model path
        self.model_path = self._find_model_path()
        
        print(f"Loading model from {self.model_path}...")
        try:
            # Check if path exists and is a directory
            if not os.path.isdir(self.model_path):
                print(f"Warning: {self.model_path} is not a directory")
                print("Looking for model files in parent directories...")
                
                # Try to find model files in parent directories
                parent_dir = os.path.dirname(self.model_path)
                if os.path.isdir(parent_dir):
                    print(f"Using parent directory: {parent_dir}")
                    self.model_path = parent_dir
            
            # Check GPU availability
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"Number of GPUs available: {torch.cuda.device_count()}")
                
                # Print CUDA version info
                cuda_version = torch.version.cuda
                print(f"CUDA Version: {cuda_version}")
                
                # Print GPU memory info
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"GPU Memory: {gpu_memory_allocated:.2f}GB allocated / {gpu_memory_total:.2f}GB total")
            else:
                self.device = torch.device("cpu")
                print("No GPU detected, falling back to CPU (this will be very slow).")
            
            # Load tokenizer
            print("Loading tokenizer...")
            try:
                # First try with trust_remote_code for newer models
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, 
                    local_files_only=True,
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"Trying alternative tokenizer loading method: {e}")
                # Try without trust_remote_code
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, 
                    local_files_only=True
                )
            
            # Load model - use device_map="auto" for GPU distribution
            print("Loading model (this may take a while)...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    load_in_8bit=True, # Try 8-bit first
                    torch_dtype=torch.float16, # Use float16 for faster inference
                    local_files_only=True,
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"Error with 8-bit quantization: {e}")
                print("Trying 4-bit quantization instead...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    load_in_4bit=True, # Try 4-bit if 8-bit fails
                    torch_dtype=torch.float16,
                    local_files_only=True,
                    trust_remote_code=True
                )
            
            # Report success and model configuration
            print("Model loaded successfully on detected devices.")
            print(f"Model name: {self.model.config._name_or_path}")
            print(f"Model parameters: {self.model.num_parameters() / 1e9:.2f}B")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to CPU mode (this will be very slow)...")
            try:
                # Fallback to CPU loading if GPU loading fails
                self.device = torch.device("cpu")
                print("Attempting to load tokenizer on CPU...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, 
                    local_files_only=True
                )
                
                print("Attempting to load model on CPU (this will likely fail with 70B models)...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    device_map="cpu",
                    local_files_only=True
                )
                print("Model loaded on CPU.")
            except Exception as fallback_e:
                print(f"Critical error loading model: {fallback_e}")
                # Exit if model cannot be loaded at all
                raise fallback_e
    
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
                # Cast to int to prevent TypeError with float values
                l_shape_grid.set_color(j, i, list(Color)[int(grid[i, j])])
        
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
    
    def _generate_with_llama(self, prompt, retries=0):
        """Generate code using Llama model via Transformers with retries."""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Track generation time
            start_time = time.time()
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95
                )
            
            # Calculate generation time
            generation_time = time.time() - start_time
            tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
            tokens_per_second = tokens_generated / generation_time
            
            print(f"Generation stats: {tokens_generated} tokens in {generation_time:.2f}s ({tokens_per_second:.2f} tokens/sec)")
            
            # Log GPU memory usage if available
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3    # Convert to GB
                print(f"GPU Memory: {gpu_memory_allocated:.2f}GB allocated, {gpu_memory_reserved:.2f}GB reserved")
            
            # Decode the output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response (everything after the prompt)
            response = generated_text[len(prompt):]
            
            return response
            
        except Exception as e:
            if retries < self.max_retries:
                print(f"Error generating code: {e}, retrying...")
                time.sleep(self.retry_delay)
                return self._generate_with_llama(prompt, retries + 1)
            else:
                print(f"Error generating code after {self.max_retries} retries: {e}")
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
    
    def _evaluate_function(self, code, grid_size):
        """Evaluate a generated function."""
        try:
            # Create a module to execute the code
            with tempfile.NamedTemporaryFile(suffix='.py') as temp_file:
                # Add necessary imports
                full_code = """import numpy as np
import random

""" + code
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
                grid = func(grid_size)
                
                if not isinstance(grid, np.ndarray) or grid.shape != (grid_size, grid_size):
                    return 0.0, None, f"Function didn't return a valid {grid_size}×{grid_size} grid"
                
                # Verify no L-shapes and calculate score
                score, points = self._verify_grid(grid)
                
                if score > 0:
                    return score, grid, "Valid solution"
                else:
                    return 0.0, None, f"Invalid solution: L-shape at {points}"
                
        except Exception as e:
            return 0.0, None, f"Error evaluating function: {str(e)}\n{traceback.format_exc()}"
    
    def _create_prompt(self, grid_size, iterations_run=0, best_solution=None):
        """Create a prompt for the Llama model."""
        system_prompt = f"""You are an expert in combinatorial optimization and Ramsey Theory.

Task: Solve the L-shape Ramsey problem for a {grid_size}×{grid_size} grid using THREE colors (0, 1, 2).
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
        if best_solution is not None:
            best_solution_text = f"""
For reference, here is a known valid solution for a {grid_size}×{grid_size} grid:
```
{best_solution}
```

This pattern is known to work. You can use it as a starting point or create an entirely new solution.
"""

        user_prompt = f"""Create a function named '{self.function_name}' that generates a valid 3-coloring for a {grid_size}×{grid_size} grid for the L-shape Ramsey problem.

The function should:
1. Take grid_size as input parameter 'n' (n={grid_size})
2. Return a numpy array with values 0, 1, 2 (representing three colors)
3. Ensure no L-shapes are monochromatic

{best_solution_text}

Iteration #{iterations_run+1}: Try to find a valid or improved solution.

Helpful hints:
1. Latin square patterns (where each row and column contains each color once) work well
2. Modular arithmetic patterns like (i + j) % 3 can also work
3. Consider patterns where adjacent cells have different colors
4. For larger grids, try to maintain color diversity in each row and column

Return ONLY the Python function without any explanation.
"""

        return f"{system_prompt}\n\n{user_prompt}"
    
    def _save_result(self, grid, score, code):
        """Save the result to disk."""
        # Create grid directory
        os.makedirs(os.path.join(self.output_dir, f"grid_{grid.shape[0]}"), exist_ok=True)
        
        # Save code
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"grid_{grid.shape[0]}", f"solution_{grid.shape[0]}_{timestamp}.py")
        
        with open(filename, "w") as f:
            f.write(f"# Solution for {grid.shape[0]}×{grid.shape[0]} grid\n")
            f.write(f"# Score: {score}\n")
            f.write(f"# Generated: {timestamp}\n\n")
            f.write("import numpy as np\n\n")
            f.write(code)
        
        # Save visualization
        vis_filename = os.path.join(self.output_dir, f"grid_{grid.shape[0]}", f"grid_{grid.shape[0]}_{timestamp}.png")
        self._visualize_grid(grid, vis_filename)
        
        print(f"Saved result to {filename}")
        print(f"Saved visualization to {vis_filename}")
    
    def _visualize_grid(self, grid, filename=None):
        """Visualize a grid."""
        plt.figure(figsize=(6, 6))
        cmap = plt.cm.colors.ListedColormap(['red', 'green', 'blue'])
        plt.imshow(grid, cmap=cmap, vmin=0, vmax=2)
        plt.grid(True, color='black', linewidth=1.5)
        plt.xticks(range(grid.shape[0]))
        plt.yticks(range(grid.shape[1]))
        plt.title(f"L-shape Ramsey {grid.shape[0]}×{grid.shape[1]} Grid Solution")
        
        if filename:
            plt.savefig(filename, dpi=150)
            plt.close()
        else:
            plt.show()
    
    def _initialize_island(self, island_idx, grid_size):
        """Initialize an island with a baseline solution."""
        print(f"Initializing island {island_idx}...")
        
        # Create prompt for initialization
        prompt = self._create_prompt(grid_size, 0, self.baseline_pattern)
        
        # Generate code
        generated_text = self._generate_with_llama(prompt)
        
        # Extract code
        code = self._extract_code(generated_text)
        
        if code:
            # Evaluate function
            score, grid, message = self._evaluate_function(code, grid_size)
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
                print("Failed to find valid solution during initialization")
        else:
            print("Failed to extract code from generation")
    
    def _evolve_island(self, island_idx, grid_size, iterations):
        """Evolve an island for a specified number of iterations."""
        print(f"Evolving island {island_idx}...")
        
        for i in range(iterations):
            print(f"\nIteration {i+1}/{iterations}...")
            
            # Create prompt for evolution
            prompt = self._create_prompt(grid_size, i, self.best_solution)
            
            # Generate code
            generated_text = self._generate_with_llama(prompt)
            
            # Extract code
            code = self._extract_code(generated_text)
            
            if code:
                # Evaluate function
                score, grid, message = self._evaluate_function(code, grid_size)
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
    
    def solve(self, grid_size, iterations=10, time_limit=300):
        """Solve the L-shape Ramsey problem for a given grid size."""
        print(f"\n{'=' * 50}")
        print(f"Solving {grid_size}×{grid_size} grid")
        print(f"{'=' * 50}")
        
        print(f"Starting FunSearch for {grid_size}×{grid_size} grid...")
        
        # Reset best solution for this grid size
        self.best_score = 0
        self.best_solution = None
        
        # Use Latin square pattern only for 3x3 grid
        if grid_size == 3:
            baseline_pattern = self._create_latin_square(3)
            baseline_score, _ = self._verify_grid(baseline_pattern)
            
            if baseline_score > 0:
                self.best_score = baseline_score
                self.best_solution = baseline_pattern
                
                print(f"Baseline Latin square solution with score {baseline_score}:")
                print(self.best_solution)
        
        # Initialize islands
        num_islands = 2  # Reduced to 2 for memory reasons with large model
        for island_idx in range(num_islands):
            self._initialize_island(island_idx, grid_size)
        
        # Evolve islands
        for island_idx in range(num_islands):
            self._evolve_island(island_idx, grid_size, iterations)
        
        # Final results
        print(f"\nFunSearch completed for {grid_size}×{grid_size} grid")
        if self.best_solution is not None:
            print(f"Best solution found (score: {self.best_score}):")
            print(self.best_solution)
        else:
            print(f"No valid solution found for {grid_size}×{grid_size} grid")
        
        return self.best_solution, self.best_score

def main():
    parser = argparse.ArgumentParser(description='FunSearch for L-shape Ramsey problem')
    parser.add_argument('--min-grid-size', type=int, default=3,
                      help='Minimum grid size to solve (default: 3)')
    parser.add_argument('--max-grid-size', type=int, default=20,
                      help='Maximum grid size to solve (default: 20)')
    parser.add_argument('--grid-size', type=int, default=None,
                      help='Specific grid size to solve (overrides min/max)')
    parser.add_argument('--iterations', type=int, default=5,
                      help='Maximum iterations per island (default: 5)')
    parser.add_argument('--time-limit', type=int, default=300,
                      help='Time limit in seconds (default: 300)')
    parser.add_argument('--model-path', type=str, default=None,
                      help='Path to Llama model (default: search in standard locations)')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for generation (default: 0.7)')
    args = parser.parse_args()
    
    # Create FunSearch instance
    funsearch = LlamaFunSearch(
        model_path=args.model_path, 
        temperature=args.temperature
    )
    
    # Determine grid sizes to solve
    if args.grid_size is not None:
        grid_sizes = [args.grid_size]
    else:
        grid_sizes = range(args.min_grid_size, args.max_grid_size + 1)
    
    # Solve for each grid size
    results = {}
    for grid_size in grid_sizes:
        print(f"\n{'#' * 70}")
        print(f"# Starting search for {grid_size}×{grid_size} grid")
        print(f"{'#' * 70}")
        
        solution, score = funsearch.solve(grid_size, args.iterations, args.time_limit)
        results[grid_size] = (solution, score)
    
    # Print summary of results
    print("\n" + "=" * 50)
    print("SUMMARY OF RESULTS")
    print("=" * 50)
    for grid_size, (solution, score) in results.items():
        if solution is not None:
            print(f"{grid_size}×{grid_size}: Valid solution found! Score: {score:.2f}")
        else:
            print(f"{grid_size}×{grid_size}: No valid solution found")
    print("=" * 50)

if __name__ == "__main__":
    main()