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

# Force GPU device settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Explicitly set to first GPU
    print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
else:
    print("❌ No GPU detected, using CPU")

# Import GGUF model support
try:
    import llama_cpp
    print(f"Using llama-cpp-python v{llama_cpp.__version__}")
    # Check for CUDA support in llama-cpp-python
    has_cuda = False
    for attr in dir(llama_cpp.llama_cpp):
        if "CUDA" in attr or "cuda" in attr or "GPU" in attr:
            has_cuda = True
            print(f"✅ CUDA support found in llama-cpp-python: {attr}")
    if not has_cuda:
        print("⚠️ No CUDA support detected in llama-cpp-python")
except ImportError:
    print("llama-cpp-python not available")

# Import other necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM

from l_shape_ramsey import LShapeGrid, Color


class LlamaFunSearch:
    """
    A FunSearch implementation for the L-shape Ramsey problem using Llama via GGUF or Transformers.
    This implementation focuses on the 3×3 grid size with the verified Latin square pattern.
    """
    
    def __init__(self, model_path=None, temperature=0.7, max_tokens=2048):
        """Initialize the LlamaFunSearch."""
        # Check for GGUF models first
        if model_path is None:
            # Look for models in different directories
            model_locations = [
                "./models/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
                "./llama_gguf/tinyllama-1.1b-q4.gguf",
                "./llama3_hf"
            ]
            
            for location in model_locations:
                if os.path.exists(location):
                    model_path = location
                    print(f"Found model: {model_path}")
                    break
        
        self.model_path = model_path
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
    
    def _load_model(self):
        """Load the Llama model using the best available method with GPU acceleration."""
        print(f"Loading model from {self.model_path}...")
        
        # Check if we're using a GGUF file
        is_gguf = self.model_path and self.model_path.lower().endswith('.gguf')
        
        try:
            # GPU detection
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                self.device = torch.device("cuda")
                print(f"🚀 GPU will be used: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                print("⚠️ No GPU detected, using CPU")
            
            # First try to load GGUF model if the path ends with .gguf
            if is_gguf and os.path.exists(self.model_path):
                try:
                    import llama_cpp
                    
                    # Basic parameters for model loading
                    model_params = {
                        "model_path": self.model_path,
                        "n_ctx": 2048,       # Context window
                        "n_threads": 4,       # CPU threads
                        "verbose": True       # Show detailed logs
                    }
                    
                    # Add GPU acceleration parameters if CUDA is available
                    if cuda_available:
                        model_params.update({
                            "n_gpu_layers": -1,        # Use all layers on GPU
                            "main_gpu": 0,             # Primary GPU device
                            "tensor_split": [1.0],     # Use all on first GPU
                            "n_batch": 512             # Batch size for inference
                        })
                    
                    # Initialize the model
                    print(f"Loading GGUF model: {self.model_path}")
                    self.model = llama_cpp.Llama(**model_params)
                    self.tokenizer = None  # Not needed for llama-cpp
                    self._model_type = "llama-cpp"
                    print("✅ GGUF model loaded successfully")
                    return
                except Exception as e:
                    print(f"⚠️ Error loading GGUF model: {str(e)}")
            
            # Try HuggingFace model
            if os.path.exists(self.model_path) and os.path.isdir(self.model_path):
                if os.path.exists(os.path.join(self.model_path, "config.json")):
                    try:
                        print(f"Loading HuggingFace model: {self.model_path}")
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.model_path,
                            trust_remote_code=True,
                            use_fast=False
                        )
                        
                        # Load model with GPU acceleration if available
                        if cuda_available:
                            self.model = AutoModelForCausalLM.from_pretrained(
                                self.model_path,
                                device_map="auto",
                                load_in_8bit=True,
                                torch_dtype=torch.float16,
                                trust_remote_code=True
                            )
                        else:
                            self.model = AutoModelForCausalLM.from_pretrained(
                                self.model_path,
                                device_map="cpu",
                                trust_remote_code=True
                            )
                        self._model_type = "huggingface"
                        print("✅ HuggingFace model loaded successfully")
                        return
                    except Exception as e:
                        print(f"⚠️ Error loading HuggingFace model: {str(e)}")
            
            # If we still don't have a model, try the default TinyLlama
            gguf_default = "./llama_gguf/tinyllama-1.1b-q4.gguf"
            if os.path.exists(gguf_default):
                try:
                    import llama_cpp
                    print(f"Loading default TinyLlama model: {gguf_default}")
                    
                    # Basic parameters
                    model_params = {
                        "model_path": gguf_default,
                        "n_ctx": 2048,
                        "n_threads": 4,
                        "verbose": True
                    }
                    
                    # Add GPU acceleration if available
                    if cuda_available:
                        model_params.update({
                            "n_gpu_layers": -1,
                            "main_gpu": 0
                        })
                    
                    self.model = llama_cpp.Llama(**model_params)
                    self.tokenizer = None
                    self._model_type = "llama-cpp"
                    print("✅ Default TinyLlama model loaded successfully")
                    return
                except Exception as e:
                    print(f"⚠️ Error loading default model: {str(e)}")
            
            # Last resort: dummy model
            print("⚠️ All model loading attempts failed. Creating a dummy model for testing.")
            self.tokenizer = None
            self.model = None
            self._model_type = "dummy"
            self._dummy_mode = True
            print("🔄 Dummy mode enabled - will use rule-based grid generation instead of LLM")
            
        except Exception as e:
            print(f"⚠️ Error during model loading: {str(e)}")
            print("🔄 Using dummy mode for testing")
            self.tokenizer = None
            self.model = None
            self._model_type = "dummy"
            self._dummy_mode = True
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
    
    def _load_model(self):
        """Load the Llama model using the best available method for GPU acceleration."""
        print(f"Loading model...")
        
        # Set explicit model path to the GGUF file we know exists
        gguf_model_path = "./models/Llama-3.3-70B-Instruct-Q4_K_M.gguf"
        
        if not os.path.exists(gguf_model_path):
            print(f"❌ Could not find model at {gguf_model_path}")
            print("Downloading model from internet...")
            # Call the model downloader script
            try:
                import subprocess
                subprocess.run(["python", "download_llama_gguf.py"], check=True)
                print("✅ Model downloaded successfully")
            except Exception as e:
                print(f"❌ Failed to download model: {e}")
        
        # Try to load the GGUF model with GPU acceleration 
        if os.path.exists(gguf_model_path):
            print(f"📂 Using GGUF model: {gguf_model_path}")
            try:
                # Import llama-cpp-python
                import llama_cpp
                
                # Check CUDA availability through torch first
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
                    # Set explicit CUDA environment variables
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                    
                    # Configure model parameters with GPU settings
                    model_params = {
                        "model_path": gguf_model_path,
                        "n_ctx": 2048,                # Context window size
                        "n_batch": 512,               # Batch size for prompt processing
                        "n_gpu_layers": -1,           # Use all layers on GPU
                        "main_gpu": 0,                # Primary GPU device
                        "verbose": True               # Show detailed logs
                    }
                    
                    # Load model with GPU acceleration
                    print("🔥 Loading model with CUDA acceleration...")
                    self.model = llama_cpp.Llama(**model_params)
                    self._model_type = "llama-cpp-gpu"
                    self.tokenizer = None  # Not needed for llama-cpp
                    print("✨ Model loaded successfully with GPU acceleration!")
                    return
                else:
                    print("⚠️ No CUDA devices detected, falling back to CPU")
            except ImportError:
                print("⚠️ llama_cpp module not available, trying alternative methods")
            except Exception as e:
                print(f"❌ Error loading GGUF model: {str(e)}")
        
        # If we couldn't load the GGUF model, try the HF directory
        hf_model_path = "./llama3_hf"
        if os.path.exists(hf_model_path) and os.path.exists(os.path.join(hf_model_path, "config.json")):
            print(f"📂 Found HuggingFace model at {hf_model_path}")
            try:
                # Load tokenizer
                print("🔄 Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    hf_model_path,
                    trust_remote_code=True,
                    use_fast=False
                )
                
                # Load model with GPU acceleration
                print("🔄 Loading model to GPU...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    hf_model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    trust_remote_code=True
                )
                self._model_type = "huggingface"
                print("✨ HuggingFace model loaded successfully on GPU!")
                return
            except Exception as e:
                print(f"❌ Error loading HuggingFace model: {str(e)}")
        
        # If we reach here, all attempts failed
        print("⚠️ Using dummy model for testing - this is NOT on GPU")
        self.tokenizer = None
        self.model = None
        self._model_type = "dummy"
        self._dummy_mode = True
    
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
    
    def _generate_with_llama(self, prompt, retries=0):
        """Generate code using Llama model."""
        # If in dummy mode, use rule-based generation
        if hasattr(self, '_dummy_mode') and self._dummy_mode:
            return self._generate_dummy_solution(prompt)

        try:
            # Generate based on the model type
            if self._model_type == "llama-cpp":
                # Use llama-cpp-python
                output = self.model(
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=0.95,
                    top_k=50
                )
                return output["choices"][0]["text"]
                
            elif self._model_type == "ctransformers":
                # Use ctransformers
                return self.model(
                    prompt=prompt,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=0.95,
                    top_k=50
                )
                
            elif self._model_type == "huggingface":
                # Use HuggingFace transformers
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95
                    )
                
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
                
    def _generate_dummy_solution(self, prompt):
        """Generate rule-based solutions when no model is available."""
        # Extract grid size from prompt
        try:
            grid_size = int([line for line in prompt.split('\n') if 'n=' in line][0].split('n=')[1].split(')')[0])
        except:
            grid_size = 3  # Default size
            
        print(f"Generating rule-based solution for {grid_size}×{grid_size} grid...")
            
        # Generate different solution types based on grid size
        solution_type = random.choice(["latin", "modular", "diagonal"])
        
        if solution_type == "latin":
            code = f"""
def generate_grid(n):
    # Latin square pattern
    grid = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            grid[i, j] = (i + j) % 3
    return grid
"""
        elif solution_type == "modular":
            code = f"""
def generate_grid(n):
    # Modular arithmetic pattern
    grid = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            grid[i, j] = (i*2 + j) % 3
    return grid
"""
        else:  # diagonal
            code = f"""
def generate_grid(n):
    # Diagonal pattern
    grid = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            # Different patterns on diagonals
            if i == j:
                grid[i, j] = 0
            elif i > j:
                grid[i, j] = 1
            else:
                grid[i, j] = 2
    return grid
"""

        # Return with Python code block format
        return f"```python\n{code}\n```"
    
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
        
        # Set our baseline solution
        baseline_score, _ = self._verify_grid(self.baseline_pattern)
        self.best_score = baseline_score
        self.best_solution = self.baseline_pattern
        
        print(f"Baseline Latin square solution with score {baseline_score}:")
        print(self.baseline_pattern)
        
        # Initialize islands
        num_islands = 2  # Reduced to 2 for memory reasons with large model
        for island_idx in range(num_islands):
            self._initialize_island(island_idx, grid_size)
        
        # Evolve islands
        for island_idx in range(num_islands):
            self._evolve_island(island_idx, grid_size, iterations)
        
        # Final results
        print(f"\nFunSearch completed for {grid_size}×{grid_size} grid")
        print(f"Best solution found (score: {self.best_score}):")
        print(self.best_solution)
        
        return self.best_solution, self.best_score

def main():
    parser = argparse.ArgumentParser(description='FunSearch for L-shape Ramsey problem')
    parser.add_argument('--grid-size', type=int, default=3,
                      help='Grid size to solve (default: 3)')
    parser.add_argument('--iterations', type=int, default=10,
                      help='Maximum iterations per island (default: 10)')
    parser.add_argument('--time-limit', type=int, default=300,
                      help='Time limit in seconds (default: 300)')
    parser.add_argument('--model-path', type=str, default=None,
                      help='Path to Llama model (default: search in standard locations)')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for generation (default: 0.7)')
    args = parser.parse_args()
    
    # Create and run FunSearch
    funsearch = LlamaFunSearch(
        model_path=args.model_path, 
        temperature=args.temperature
    )
    
    funsearch.solve(args.grid_size, args.iterations, args.time_limit)

if __name__ == "__main__":
    main()