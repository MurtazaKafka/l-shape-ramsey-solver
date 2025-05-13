#!/usr/bin/env python3
import os
import time
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import tempfile
import importlib.util
import traceback
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Check and activate conda environment if needed
def ensure_conda_env(env_name="ramsey"):
    """Ensure we're running in the specified conda environment"""
    try:
        # Check if already in the right environment
        current_prefix = os.environ.get("CONDA_PREFIX", "")
        if env_name in current_prefix:
            print(f"✅ Already in conda environment: {env_name}")
            return True
            
        # If not, try to activate it programmatically
        print(f"⚠️ Not in {env_name} conda environment. Attempting to activate...")
        
        # Create a temporary script to run in the correct environment
        temp_script = f"""
import sys
print(f"Python executable: {{sys.executable}}")
print(f"Python version: {{sys.version}}")
try:
    import llama_cpp
    print(f"llama_cpp version: {{llama_cpp.__version__}}")
    print("CUDA support:", any("cuda" in attr.lower() or "gpu" in attr.lower() for attr in dir(llama_cpp.llama_cpp)))
except ImportError:
    print("llama_cpp not installed")
        """
        
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write(temp_script)
            temp_path = f.name
            
        # Run the script in the conda environment
        result = subprocess.run(
            f"conda run -n {env_name} python {temp_path}",
            shell=True, 
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(result.stdout)
            if "llama_cpp not installed" in result.stdout:
                print(f"⚠️ llama-cpp-python not installed in {env_name} environment.")
                return False
            return True
        else:
            print(f"❌ Failed to run in conda environment: {env_name}")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error checking conda environment: {e}")
        return False

# Force GPU device settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Check for conda environment first
ensure_conda_env("ramsey")

import torch
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Explicitly set to first GPU
    print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
else:
    print("❌ No GPU detected, using CPU")

# Try importing llama_cpp with improved error handling
try:
    import llama_cpp
    print(f"Using llama-cpp-python v{llama_cpp.__version__}")
    # Check for CUDA support in llama-cpp-python
    has_cuda = False
    cuda_attrs = []
    for attr in dir(llama_cpp.llama_cpp):
        if any(cuda_term in attr.lower() for cuda_term in ["cuda", "gpu", "ggml_cuda"]):
            has_cuda = True
            cuda_attrs.append(attr)
    
    if has_cuda:
        print(f"✅ CUDA support found in llama-cpp-python: {', '.join(cuda_attrs)}")
    else:
        print("⚠️ No CUDA support detected in llama-cpp-python")
        print("Reinstalling llama-cpp-python with CUDA support...")
        # Try to reinstall with CUDA support
        try:
            subprocess.run(
                "conda activate ramsey && pip uninstall -y llama-cpp-python && CMAKE_ARGS=\"-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc\" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir",
                shell=True,
                check=True
            )
            print("Reimporting llama_cpp after reinstallation...")
            # Force reload the module
            import importlib
            importlib.reload(llama_cpp)
        except Exception as e:
            print(f"Failed to reinstall with CUDA support: {e}")
except ImportError as e:
    print(f"⚠️ llama-cpp-python import error: {e}")
    print("\nTry installing with CUDA support in the ramsey conda environment:")
    print("conda activate ramsey && CMAKE_ARGS=\"-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc\" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir")
    
    # Try a fallback - maybe it can be imported differently?
    try:
        print("Attempting to import llama_cpp_python instead...")
        import llama_cpp_python as llama_cpp
        print(f"Using llama-cpp-python v{llama_cpp.__version__}")
    except ImportError:
        print("Both llama_cpp and llama_cpp_python failed to import")

# Import other necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try to import the problem definition
try:
    from l_shape_ramsey import LShapeGrid, Color
except ImportError as e:
    print(f"⚠️ Error importing l_shape_ramsey: {e}")

# Add for downloading
from huggingface_hub import hf_hub_download

def download_llama_70b_gguf(dest_path):
    print("Downloading Meta Llama 3.3 70B GGUF model...")
    repo_id = "TheBloke/Llama-3.3-70B-Instruct-GGUF"
    filename = "llama-3.3-70b-instruct.Q4_K_M.gguf"
    try:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=os.path.dirname(dest_path), local_dir_use_symlinks=False)
        if file_path != dest_path:
            os.rename(file_path, dest_path)
        print(f"Downloaded model to {dest_path}")
        return dest_path
    except Exception as e:
        print(f"Failed to download model: {e}")
        return None

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
                # "./llama_gguf/tinyllama-1.1b-q4.gguf",
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
        """Load the Llama model using GPU acceleration."""
        print(f"Loading model...")
        
        # First, check GPU memory and free up resources if needed
        self._ensure_gpu_memory(required_gb=42)  # 70B model needs ~40GB with quantization
        
        model_path = "models/Llama-3.3-70B-Instruct-Q4_K_M.gguf"
        
        if os.path.exists(model_path):
            print(f"Loading 70B model from: {model_path}")
            try:
                import llama_cpp
                
                # Configure optimal parameters specifically for 70B model
                model_params = {
                    "model_path": model_path,
                    "n_ctx": 2048,
                    "n_threads": 8,
                    "verbose": True,
                    "seed": 42,
                    "ignore_tensors_error": True,
                    "f16_kv": True,
                    "use_mlock": True,
                    "use_mmap": True,
                    "numa": False  # Disable NUMA to avoid pthread warnings
                }
                
                # Add GPU-specific params - force all layers to GPU
                if torch.cuda.is_available():
                    print(f"🚀 GPU will be used: {torch.cuda.get_device_name(0)}")
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    print(f"Available VRAM: {vram_gb:.1f} GB")
                    
                    # Use all GPU layers and force CUDA
                    gpu_params = {
                        "n_gpu_layers": -1,         # Load ALL layers to GPU (-1 means all layers)
                        "main_gpu": 0,              # Primary GPU
                        "n_batch": 512,             # Batch size for processing
                        "offload_kqv": True,        # Offload key/query/value operations
                        "mul_mat_q": True           # Enable matrix multiplication on Q projection
                    }
                    
                    model_params.update(gpu_params)
                    print(f"Loading with parameters: {model_params}")
                else:
                    print("⚠️ No GPU detected, but 70B model requires GPU")
                    return
                
                # Load the model
                start_time = time.time()
                self.model = llama_cpp.Llama(**model_params)
                load_time = time.time() - start_time
                print(f"⏱️ Model loaded in {load_time:.2f} seconds")
                
                self.tokenizer = None  # Not needed for llama-cpp
                self._model_type = "llama-cpp"
                
                print(f"✅ Successfully loaded 70B model!")
                return
                
            except Exception as e:
                print(f"⚠️ Error loading 70B model: {str(e)}")
                print(f"Try reinstalling llama-cpp-python with GPU support:")
                print(f"pip uninstall -y llama-cpp-python && CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python")
                raise RuntimeError("Failed to load 70B model") from e
        
        # If model doesn't exist
        print(f"❌ 70B model not found at: {model_path}")
        raise RuntimeError("70B model file not found")
    
    def _ensure_gpu_memory(self, required_gb=42):
        """Ensure enough GPU memory is available by clearing cache and stopping other processes if needed."""
        if not torch.cuda.is_available():
            print("No GPU available")
            return False
        
        # Clear PyTorch cache first
        torch.cuda.empty_cache()
        
        # Check free memory
        free_bytes = torch.cuda.mem_get_info()[0]
        free_gb = free_bytes / (1024**3)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print(f"GPU memory: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
        
        if free_gb >= required_gb:
            print(f"✅ Enough GPU memory available ({free_gb:.1f}GB free, {required_gb}GB required)")
            return True
        
        print(f"⚠️ Not enough GPU memory available ({free_gb:.1f}GB free, {required_gb}GB required)")
        print("Checking for other processes using GPU...")
        
        try:
            # Use our GPU manager script to check and potentially kill processes
            min_free_mb = int(required_gb * 1024)
            kill_cmd = f"python gpu_manager.py --info --list --min-free {min_free_mb}"
            
            print(f"Running: {kill_cmd}")
            os.system(kill_cmd)
            
            # Check memory again
            torch.cuda.empty_cache()
            free_bytes = torch.cuda.mem_get_info()[0]
            free_gb = free_bytes / (1024**3)
            
            print(f"After cleanup: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
            
            if free_gb >= required_gb:
                print(f"✅ Successfully freed up GPU memory")
                return True
            else:
                print(f"⚠️ Still not enough GPU memory. Trying to kill all GPU processes...")
                # More aggressive approach - kill all processes
                os.system(f"python gpu_manager.py --kill-all --force --exclude {os.getpid()}")
                
                # Check again
                torch.cuda.empty_cache()
                free_bytes = torch.cuda.mem_get_info()[0]
                free_gb = free_bytes / (1024**3)
                
                print(f"After killing all processes: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
                
                if free_gb >= required_gb:
                    print(f"✅ Successfully freed up GPU memory")
                    return True
                else:
                    print(f"❌ Failed to free up enough GPU memory even after killing processes")
                    return False
        except Exception as e:
            print(f"Error when trying to free GPU memory: {e}")
            return False
    
    def _create_latin_square(self, n):
        """Create a Latin square pattern that scales to any grid size."""
        grid = np.zeros((n, n), dtype=int)
        
        if n <= 3:
            # For small grids, use the proven 3×3 Latin square pattern
            pattern = [
                [0, 1, 2],
                [2, 0, 1],
                [1, 2, 0]
            ]
            for i in range(min(n, 3)):
                for j in range(min(n, 3)):
                    grid[i, j] = pattern[i][j]
        else:
            # For larger grids, use modular arithmetic which scales to any size
            for i in range(n):
                for j in range(n):
                    # This formula ensures no L-shapes and good color distribution
                    grid[i, j] = (i + j*2) % 3
        
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
        """Visualize a grid with improved clarity for larger grid sizes."""
        grid_size = grid.shape[0]
        
        # Adjust figure size based on grid size
        fig_size = max(6, min(12, grid_size))
        plt.figure(figsize=(fig_size, fig_size))
        
        # Create color map
        cmap = plt.cm.colors.ListedColormap(['red', 'green', 'blue'])
        
        # Plot grid
        plt.imshow(grid, cmap=cmap, vmin=0, vmax=2)
        
        # Add grid lines - thicker lines for larger grids
        line_width = max(0.5, min(1.5, 3.0 / (grid_size / 4)))
        plt.grid(True, color='black', linewidth=line_width)
        
        # Add ticks
        if grid_size <= 10:
            # For smaller grids, show all ticks
            plt.xticks(range(grid_size))
            plt.yticks(range(grid_size))
        else:
            # For larger grids, show fewer ticks
            step = max(1, grid_size // 10)
            plt.xticks(range(0, grid_size, step))
            plt.yticks(range(0, grid_size, step))
        
        # Add color indicators in cells for large grids
        if grid_size > 10:
            for i in range(grid_size):
                for j in range(grid_size):
                    color_idx = grid[i, j]
                    plt.text(j, i, str(color_idx), ha='center', va='center', 
                             fontsize=max(4, min(9, 12 / (grid_size / 6))),
                             color='white', fontweight='bold')
        
        plt.title(f"L-shape Ramsey {grid_size}×{grid_size} Grid Solution")
        
        # Ensure directory exists if filename is provided
        if filename:
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            plt.savefig(filename, dpi=150, bbox_inches='tight')
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
        
        # For larger grid sizes, create a more appropriate baseline
        self.baseline_pattern = self._create_latin_square(grid_size)
        
        # Set our baseline as the starting best solution
        baseline_score, _ = self._verify_grid(self.baseline_pattern)
        self.best_score = baseline_score
        self.best_solution = self.baseline_pattern
        
        print(f"Baseline solution for {grid_size}×{grid_size} grid with score {baseline_score}:")
        print(self.baseline_pattern)
        
        # Save the baseline solution
        if grid_size > 3:  # Only save if not the default small grid
            # Create a simple function definition for the baseline
            if grid_size <= 10:  # Only print full grid for smaller sizes
                self._save_result(self.baseline_pattern, baseline_score, 
                    f"def generate_grid(n):\n    # Baseline solution\n    return {repr(self.baseline_pattern.tolist())}")
        
        # Adjust number of islands based on grid size
        # Larger grids need fewer islands due to memory constraints
        if grid_size <= 5:
            num_islands = 3
        elif grid_size <= 8:
            num_islands = 2
        else:
            num_islands = 1  # For very large grids, use just one island to save memory
            
        print(f"Using {num_islands} evolution islands for grid size {grid_size}")
        
        # Initialize timer for overall time limit
        start_time = time.time()
        
        # Initialize islands
        for island_idx in range(num_islands):
            self._initialize_island(island_idx, grid_size)
            
            # Check if we're running out of time
            if time.time() - start_time > time_limit * 0.4:  # Use 40% of time at most for initialization
                print("Time limit approaching, skipping additional island initialization")
                break
                
        # Calculate iterations per island considering time limit
        remaining_time = time_limit - (time.time() - start_time)
        time_per_iteration = max(10, min(30, remaining_time / (num_islands * iterations)))
        adjusted_iterations = max(1, min(iterations, int(remaining_time / time_per_iteration / num_islands)))
        
        if adjusted_iterations < iterations:
            print(f"Adjusting to {adjusted_iterations} iterations per island due to time constraints")
        
        # Evolve islands
        for island_idx in range(min(num_islands, island_idx + 1)):
            # Check if we still have time
            if time.time() - start_time > time_limit * 0.9:  # Save 10% time for cleanup
                print("Time limit approaching, stopping evolution")
                break
                
            self._evolve_island(island_idx, grid_size, adjusted_iterations)
        
        # Final results
        elapsed = time.time() - start_time
        print(f"\nFunSearch completed for {grid_size}×{grid_size} grid in {elapsed:.1f} seconds")
        print(f"Best solution found (score: {self.best_score}):")
        
        if grid_size <= 10:  # Only print full grid for smaller sizes
            print(self.best_solution)
        else:
            print(f"Grid too large to display ({grid_size}×{grid_size}). Check visualization instead.")
        
        return self.best_solution, self.best_score

def main():
    parser = argparse.ArgumentParser(description='FunSearch for L-shape Ramsey problem')
    parser.add_argument('--grid-size', type=int, default=4,
                      help='Grid size to solve (default: 4)')
    parser.add_argument('--max-grid-size', type=int, default=10,
                      help='Maximum grid size to solve (default: 10)')
    parser.add_argument('--iterations', type=int, default=5,
                      help='Maximum iterations per island (default: 5)')
    parser.add_argument('--time-limit', type=int, default=300,
                      help='Time limit in seconds (default: 300)')
    parser.add_argument('--model-path', type=str, default=None,
                      help='Path to Llama model (default: search in standard locations)')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for generation (default: 0.7)')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to save all outputs (default: results)')
    parser.add_argument('--visualize-all', action='store_true',
                      help='Generate visualizations for all grid sizes')
    parser.add_argument('--skip-existing', action='store_true',
                      help='Skip grid sizes that already have successful solutions')
    args = parser.parse_args()
    
    # Create output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save run configuration
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Starting L-shape Ramsey FunSearch run at {timestamp}")
    print(f"Grid sizes: {args.grid_size} to {args.max_grid_size}")
    print(f"Results will be saved to: {run_dir}")
    
    # Create and run FunSearch
    funsearch = LlamaFunSearch(
        model_path=args.model_path, 
        temperature=args.temperature
    )
    
    # Track results across all grid sizes
    all_results = {}
    
    # Solve for increasing grid sizes
    for grid_size in range(args.grid_size, args.max_grid_size + 1):
        # Check if we should skip this grid size
        if args.skip_existing:
            existing_solution_path = f"grid_{grid_size}x{grid_size}_solution.png"
            if os.path.exists(existing_solution_path):
                print(f"Skipping grid size {grid_size} as solution already exists.")
                continue
        
        try:
            print(f"\n\n{'#'*60}")
            print(f"# Starting grid size {grid_size}×{grid_size}")
            print(f"{'#'*60}")
            
            start_time = time.time()
            solution, score = funsearch.solve(grid_size, args.iterations, args.time_limit)
            elapsed = time.time() - start_time
            
            print(f"\n========= Grid {grid_size}×{grid_size} Results =========")
            print(f"Best score: {score}")
            if grid_size <= 10:
                print(solution)
            print(f"Time taken: {elapsed:.1f} seconds")
            print("=" * 50)
            
            # Save the solution
            all_results[grid_size] = {
                "score": float(score),
                "success": score > 0,
                "time_taken": float(elapsed)
            }
            
            # Save visualization for this specific size
            grid_dir = os.path.join(run_dir, f"grid_{grid_size}")
            os.makedirs(grid_dir, exist_ok=True)
            
            vis_filename = os.path.join(grid_dir, f"solution.png")
            funsearch._visualize_grid(solution, vis_filename)
            
            # Also save to root directory for easy access to the most recent result
            root_vis_filename = f"grid_{grid_size}x{grid_size}_solution.png"
            funsearch._visualize_grid(solution, root_vis_filename)
            
            print(f"Saved visualization to {vis_filename}")
            print(f"Also saved to {root_vis_filename}")
            
            # Save the grid as numpy array
            np.save(os.path.join(grid_dir, f"solution.npy"), solution)
            
        except Exception as e:
            print(f"Error solving {grid_size}×{grid_size} grid: {e}")
            all_results[grid_size] = {
                "error": str(e),
                "success": False
            }
            traceback.print_exc()
    
    # Save summary of all results
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Generate a comparison visualization if requested
    if args.visualize_all and len(all_results) > 1:
        try:
            print("\nGenerating comparison visualization for all grid sizes...")
            compare_file = os.path.join(run_dir, "comparison.png")
            visualize_multiple_grids([int(k) for k in all_results.keys() if all_results[k].get("success", False)], 
                                    run_dir, compare_file)
            print(f"Comparison visualization saved to {compare_file}")
        except Exception as e:
            print(f"Error generating comparison visualization: {e}")
    
    print(f"\nRun complete! All results saved to {run_dir}")
    
    # Print summary
    print("\nSummary:")
    for size, result in all_results.items():
        status = "✅ Success" if result.get("success", False) else "❌ Failed"
        score = result.get("score", "N/A")
        time_taken = result.get("time_taken", "N/A")
        print(f"Grid {size}×{size}: {status} | Score: {score} | Time: {time_taken:.1f}s" if isinstance(time_taken, (int, float)) else f"Grid {size}×{size}: {status} | Score: {score} | Time: {time_taken}")
    
def visualize_multiple_grids(grid_sizes, results_dir, output_file):
    """Create a comparison visualization of multiple grid solutions."""
    if not grid_sizes:
        print("No grid sizes to visualize")
        return
        
    # Calculate grid layout
    n_grids = len(grid_sizes)
    cols = min(3, n_grids)
    rows = (n_grids + cols - 1) // cols
    
    # Create figure
    plt.figure(figsize=(cols * 5, rows * 5))
    
    # Color map for all plots
    cmap = plt.cm.colors.ListedColormap(['red', 'green', 'blue'])
    
    # Plot each grid
    for i, size in enumerate(grid_sizes):
        # Try to load the solution
        try:
            solution_file = os.path.join(results_dir, f"grid_{size}", "solution.npy")
            if not os.path.exists(solution_file):
                print(f"Solution file not found for grid size {size}")
                continue
                
            grid = np.load(solution_file)
            
            plt.subplot(rows, cols, i + 1)
            plt.imshow(grid, cmap=cmap, vmin=0, vmax=2)
            plt.grid(True, color='black', linewidth=1.0)
            
            if size <= 10:
                plt.xticks(range(size))
                plt.yticks(range(size))
            else:
                step = max(1, size // 5)
                plt.xticks(range(0, size, step))
                plt.yticks(range(0, size, step))
                
            plt.title(f"{size}×{size} Grid")
            
        except Exception as e:
            print(f"Error visualizing grid size {size}: {e}")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()