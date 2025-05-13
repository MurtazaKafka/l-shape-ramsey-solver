#!/usr/bin/env python3
"""
Script to fix NumPy compatibility issues and run llama_funsearch.py with proper GPU support.
This script:
1. Installs the correct NumPy version (1.26.4) that's compatible with matplotlib
2. Verifies the environment is properly set up for GPU acceleration
3. Runs the llama_funsearch.py script with GPU support
"""
import os
import sys
import subprocess
import time

def run_command(cmd, check=True):
    """Run a command and print its output."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def setup_environment():
    """Set up the environment for GPU acceleration with proper NumPy version."""
    print("Setting up environment for GPU acceleration...")
    
    # Step 1: Install the correct NumPy version
    print("\n1️⃣ Installing NumPy 1.26.4 (compatible with matplotlib)...")
    success = run_command("pip install numpy==1.26.4 --force-reinstall")
    if not success:
        print("❌ Failed to install NumPy 1.26.4")
        return False
    
    # Step 2: Set environment variables for GPU memory optimization
    print("\n2️⃣ Setting GPU memory optimization environment variables...")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Step 3: Verify CUDA is available in PyTorch
    print("\n3️⃣ Verifying CUDA is available in PyTorch...")
    verify_cmd = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
"""
    with open("verify_cuda.py", "w") as f:
        f.write(verify_cmd)
    
    success = run_command("python verify_cuda.py")
    if not success:
        print("❌ Failed to verify CUDA availability")
        return False
    
    # Step 4: Verify llama-cpp-python has CUDA support
    print("\n4️⃣ Verifying llama-cpp-python has CUDA support...")
    verify_cmd = """
try:
    import llama_cpp
    print(f"llama-cpp-python version: {llama_cpp.__version__}")
    
    # Check if CUDA is available in llama-cpp-python
    has_cuda = False
    for attr in dir(llama_cpp.llama_cpp):
        if "CUDA" in attr or "cuda" in attr or "GPU" in attr:
            has_cuda = True
            print(f"Found CUDA indicator: {attr}")
    
    if has_cuda:
        print("✅ llama-cpp-python has CUDA support")
    else:
        print("⚠️ No CUDA support found in llama-cpp-python")
except ImportError:
    print("❌ llama-cpp-python not installed")
"""
    with open("verify_llama_cpp.py", "w") as f:
        f.write(verify_cmd)
    
    success = run_command("python verify_llama_cpp.py")
    if not success:
        print("⚠️ Could not verify llama-cpp-python CUDA support")
    
    # Step 5: Verify NumPy and matplotlib compatibility
    print("\n5️⃣ Verifying NumPy and matplotlib compatibility...")
    verify_cmd = """
import numpy as np
print(f"NumPy version: {np.__version__}")
import matplotlib
print(f"Matplotlib version: {matplotlib.__version__}")
import matplotlib.pyplot as plt
print("✅ Successfully imported matplotlib")
"""
    with open("verify_numpy_matplotlib.py", "w") as f:
        f.write(verify_cmd)
    
    success = run_command("python verify_numpy_matplotlib.py")
    if not success:
        print("❌ NumPy and matplotlib compatibility issue persists")
        return False
    
    print("\n✅ Environment setup complete!")
    return True

def modify_llama_funsearch():
    """Modify llama_funsearch.py to specifically target GPU acceleration."""
    print("\nModifying llama_funsearch.py for improved GPU support...")
    
    # Create a backup of the original file if it doesn't exist
    if not os.path.exists("llama_funsearch.py.gpu_backup"):
        with open("llama_funsearch.py", "r") as f:
            original_content = f.read()
        
        with open("llama_funsearch.py.gpu_backup", "w") as f:
            f.write(original_content)
    
    # Create a new version with GPU optimization
    gpu_code = """#!/usr/bin/env python3
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

"""
    
    # Write the GPU-optimized code to a temporary file
    with open("llama_funsearch_gpu.py", "w") as f:
        f.write(gpu_code)
    
    # Append the rest of the original file
    with open("llama_funsearch.py", "r") as f:
        # Skip the first few lines (imports)
        lines = f.readlines()
        start_copy = False
        for line in lines:
            if "class LlamaFunSearch:" in line:
                start_copy = True
                break
    
    with open("llama_funsearch_gpu.py", "a") as f:
        if start_copy:
            # Modify the _load_model method to prioritize GPU
            f.write("""
class LlamaFunSearch:
    \"\"\"
    A FunSearch implementation for the L-shape Ramsey problem using Llama via GGUF or Transformers.
    This implementation focuses on the 3×3 grid size with the verified Latin square pattern.
    \"\"\"
    
    def __init__(self, model_path=None, temperature=0.7, max_tokens=2048):
        \"\"\"Initialize the LlamaFunSearch.\"\"\"
        # Check for GGUF models first
        if model_path is None:
            # Look for models in different directories
            model_locations = [
                "./models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
                "./llama_gguf/tinyllama-1.1b-q4.gguf",
                "./models/Llama-3.3-70B-Instruct-Q4_K_M.gguf"
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
        \"\"\"Load the Llama model using the best available method with GPU acceleration.\"\"\"
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
""")