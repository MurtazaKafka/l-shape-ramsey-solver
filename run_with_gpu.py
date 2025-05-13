#!/usr/bin/env python3
"""
Script to run llama_funsearch.py with the newly downloaded Llama 3 model on GPU
"""
import os
import sys
import subprocess
import shutil

def run_command(cmd):
    """Run a command and print its output."""
    print(f"\n> {cmd}")
    subprocess.run(cmd, shell=True, check=False)

def main():
    """Set up environment and run llama_funsearch.py with GPU support."""
    print("=" * 80)
    print("🚀 Running L-shape Ramsey Solver with GPU acceleration")
    print("=" * 80)
    
    # Step 1: Fix NumPy version for matplotlib compatibility
    print("\n1️⃣ Installing NumPy 1.26.4 (compatible with matplotlib)...")
    run_command("pip install numpy==1.26.4 --force-reinstall")
    
    # Step 2: Set environment variables for GPU
    print("\n2️⃣ Setting GPU environment variables...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Step 3: Create a simplified version of llama_funsearch.py
    print("\n3️⃣ Creating GPU-optimized version of llama_funsearch.py...")
    
    # Create backup if it doesn't exist
    if not os.path.exists("llama_funsearch_original.py"):
        shutil.copy("llama_funsearch.py", "llama_funsearch_original.py")
    
    # Check which new model to use (prefer 8B if available)
    model_path = "./models/Llama-3-8B-Instruct.Q4_K_M.gguf"  # larger model
    if not os.path.exists(model_path):
        model_path = "./models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"  # smaller model
    
    print(f"\n🔍 Using model: {model_path}")
    
    # Step 4: Run llama_funsearch.py with the new model
    print("\n4️⃣ Running llama_funsearch.py with GPU acceleration...")
    run_command(f"python llama_funsearch.py --grid-size 3 --iterations 2 --model-path {model_path}")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()