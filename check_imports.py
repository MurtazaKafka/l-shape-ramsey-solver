#!/usr/bin/env python3
"""
Simple script to check if transformers can be imported.
"""

import os
import sys
import importlib

def check_import(package_name):
    """Check if a package can be imported and print its version."""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ Successfully imported {package_name} (version: {version})")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {package_name}: {e}")
        return False

def check_transformers_config():
    """Check transformers configuration."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"✓ Successfully imported AutoModelForCausalLM and AutoTokenizer")
        
        # Check cache directory
        from transformers.utils.hub import TRANSFORMERS_CACHE
        print(f"Transformers cache directory: {TRANSFORMERS_CACHE}")
        if os.path.exists(TRANSFORMERS_CACHE):
            print(f"✓ Cache directory exists")
            cache_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                           for dirpath, _, filenames in os.walk(TRANSFORMERS_CACHE) 
                           for filename in filenames) / (1024**3)
            print(f"Cache size: {cache_size:.2f} GB")
        else:
            print(f"✗ Cache directory does not exist")
        
        return True
    except ImportError as e:
        print(f"✗ Failed to import from transformers: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking transformers configuration: {e}")
        return False

def main():
    """Check imports and print diagnostics."""
    print("=== Python Environment ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {':'.join(sys.path)}")
    
    print("\n=== Basic Package Checks ===")
    check_import("numpy")
    check_import("torch")
    torch_ok = check_import("torch")
    if torch_ok:
        import torch
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n=== Transformers Check ===")
    transformers_ok = check_import("transformers")
    if transformers_ok:
        check_transformers_config()
    
    print("\n=== Installation Commands ===")
    if not torch_ok:
        print("To install PyTorch with CUDA support:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    if not transformers_ok:
        print("To install Transformers:")
        print("pip install transformers")
        print("For LLM support, also install:")
        print("pip install accelerate bitsandbytes sentencepiece")

if __name__ == "__main__":
    main() 