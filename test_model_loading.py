#!/usr/bin/env python3
"""
Test script to verify the model loading capabilities.
Use this to check if your environment is correctly set up before running the full FunSearch.
"""

import os
import sys
import argparse
import torch
import glob
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

def check_common_locations():
    """Check common locations for Llama models."""
    common_locations = [
        # Home directories
        f"{Path.home()}/.cache/huggingface/hub",
        f"{Path.home()}/models",
        f"{Path.home()}/llama",
        
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
    possible_models = []
    for location in common_locations:
        if not os.path.exists(location):
            continue
            
        # Check for potential model directories
        for pattern in ["*llama*", "*Llama*", "*LLAMA*", "*70B*"]:
            matches = glob.glob(f"{location}/{pattern}")
            possible_models.extend(matches)
    
    if possible_models:
        print("Found potential Llama model locations:")
        for model in possible_models:
            print(f"  {model}")
        return possible_models
    
    return []

def test_model_loading(model_path, load_model=False):
    """Test if the model can be found and loaded."""
    original_path = model_path
    print(f"Testing model loading from: {model_path}")
    
    # Check if path exists
    if not os.path.exists(model_path):
        print(f"ERROR: Path {model_path} does not exist")
        
        # Try parent directory
        parent_dir = os.path.dirname(model_path)
        if os.path.exists(parent_dir):
            print(f"Trying parent directory: {parent_dir}")
            model_path = parent_dir
        else:
            # Check common locations
            print("Checking common model locations...")
            possible_models = check_common_locations()
            
            if possible_models:
                # Try the first found model
                model_path = possible_models[0]
                print(f"Trying alternative model path: {model_path}")
            else:
                print("No alternative model paths found.")
                return False
    
    # Get directory info
    if os.path.isdir(model_path):
        print(f"Path is a directory")
        dir_contents = os.listdir(model_path)
        print(f"Directory contains {len(dir_contents)} files/folders")
        print(f"Sample contents: {dir_contents[:5]}")
        
        # Check for model files
        model_files = [f for f in dir_contents if f.endswith('.bin') or f in ['config.json', 'tokenizer.json']]
        if model_files:
            print(f"Found model files: {', '.join(model_files)}")
        else:
            # If no model files, check subdirectories
            print("No model files found in directory, checking subdirectories...")
            
            # Look for subdirectories that might contain the model
            subdirs = [os.path.join(model_path, d) for d in dir_contents if os.path.isdir(os.path.join(model_path, d))]
            for subdir in subdirs:
                if os.path.isdir(subdir):
                    subdir_contents = os.listdir(subdir)
                    model_files = [f for f in subdir_contents if f.endswith('.bin') or f in ['config.json', 'tokenizer.json']]
                    if model_files:
                        print(f"Found model files in {subdir}: {', '.join(model_files)}")
                        model_path = subdir
                        break
    else:
        print(f"Path is a file, checking parent directory")
        parent_dir = os.path.dirname(model_path)
        if os.path.isdir(parent_dir):
            print(f"Parent directory: {parent_dir}")
            dir_contents = os.listdir(parent_dir)
            print(f"Parent directory contains {len(dir_contents)} files/folders")
            print(f"Sample contents: {dir_contents[:5]}")
            model_path = parent_dir
    
    # Check GPU
    print("\nChecking GPU availability:")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✓ GPU available! Found {device_count} device(s)")
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({memory:.1f} GB)")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("✗ No GPU detected! This will be very slow if it works at all.")
    
    # Check transformers version
    try:
        import transformers
        print(f"\nTransformers version: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not installed! Please install with: pip install transformers")
        return False
    
    # Try loading tokenizer
    print("\nAttempting to load tokenizer...")
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            local_files_only=True,
            trust_remote_code=True
        )
        print("✓ Tokenizer loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        print("Trying alternative method...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                local_files_only=True
            )
            print("✓ Tokenizer loaded successfully with alternative method!")
        except Exception as e2:
            print(f"✗ Error loading tokenizer with alternative method: {e2}")
            return False
    
    # Try loading model if requested
    if load_model:
        print("\nAttempting to load model (this may take a while)...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                load_in_8bit=True,
                torch_dtype=torch.float16,
                local_files_only=True,
                trust_remote_code=True
            )
            print("✓ Model loaded successfully!")
            print(f"Model name: {model.config._name_or_path}")
            print(f"Model size: {model.num_parameters() / 1e9:.2f}B parameters")
            
            # Test simple generation
            if tokenizer:
                print("\nTesting simple generation...")
                inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=20)
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Generated: {generated_text}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("Trying 4-bit quantization instead...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    load_in_4bit=True,
                    torch_dtype=torch.float16,
                    local_files_only=True,
                    trust_remote_code=True
                )
                print("✓ Model loaded successfully with 4-bit quantization!")
            except Exception as e2:
                print(f"✗ Error loading model with 4-bit quantization: {e2}")
                return False
    
    # If model path changed, suggest updating scripts
    if model_path != original_path and tokenizer is not None:
        print(f"\nNOTE: Successfully loaded model from {model_path} instead of {original_path}")
        print("You should update your scripts with the correct path:")
        print(f"python update_model_path.py {model_path}")
    
    print("\nAll tests completed!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Test model loading')
    parser.add_argument('--model-path', type=str, 
                        default="/home/DAVIDSON/murtaza/.llama/checkpoints/Llama3.3-70B-Instruct",
                        help='Path to the model directory')
    parser.add_argument('--load-model', action='store_true',
                        help='Actually load the model (may be slow)')
    args = parser.parse_args()
    
    success = test_model_loading(args.model_path, args.load_model)
    
    if success:
        print("\n✓ Model loading test passed!")
        print("You can now run the main script: python llama_funsearch.py")
        sys.exit(0)
    else:
        print("\n✗ Model loading test failed! Please check the error messages above.")
        print("\nIf you know the correct path to the model, run:")
        print("python update_model_path.py /correct/path/to/model")
        print("\nOr use search_model.py to find potential model locations:")
        print("python search_model.py")
        sys.exit(1)

if __name__ == "__main__":
    main() 