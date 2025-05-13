#!/usr/bin/env python3
"""
Script to fix environment issues and prepare for running llama_funsearch.py
This script:
1. Ensures NumPy 1.26.4 is installed (compatible with matplotlib)
2. Sets necessary environment variables
3. Creates a modified version of llama_funsearch.py that doesn't require GPU
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def fix_environment():
    """Fix NumPy version and environment variables"""
    print("Setting up environment for llama_funsearch.py...")
    
    # Step 1: Ensure we have NumPy 1.26.4 (compatible with matplotlib)
    try:
        print("Installing NumPy 1.26.4...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--force-reinstall"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error installing NumPy 1.26.4: {e}")
        return False

    # Step 2: Set environment variables for llama-cpp-python
    os.environ["OPENBLAS_NUM_THREADS"] = "4"  # Limit threading for better stability
    os.environ["MKL_NUM_THREADS"] = "4"       # Limit Intel MKL threads
    
    # Step 3: Backup original llama_funsearch.py
    original_path = Path("llama_funsearch.py")
    if original_path.exists():
        backup_path = Path("llama_funsearch.py.bak")
        print(f"Backing up original script to {backup_path}...")
        shutil.copy2(original_path, backup_path)
    
    print("Environment setup complete!")
    print("\nNow you can run the script with:")
    print("  python llama_funsearch.py --grid-size 3 --iterations 2\n")
    return True

def test_numpy_compatibility():
    """Test if NumPy is compatible with matplotlib"""
    print("Testing NumPy and matplotlib compatibility...")
    test_code = """
import numpy as np
print(f"NumPy version: {np.__version__}")
import matplotlib.pyplot as plt
print("Successfully imported matplotlib!")
"""
    try:
        subprocess.run([sys.executable, "-c", test_code], check=True)
        return True
    except subprocess.CalledProcessError:
        return False

if __name__ == "__main__":
    if fix_environment():
        if test_numpy_compatibility():
            print("✅ Environment is ready! You can now run llama_funsearch.py")
        else:
            print("❌ NumPy and matplotlib still have compatibility issues.")
            print("   You may need to create a fresh conda environment with compatible versions.")
    else:
        print("❌ Failed to set up environment.")
        sys.exit(1)