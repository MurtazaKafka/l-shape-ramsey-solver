#!/usr/bin/env python3
"""
Test script to verify the model loading capabilities.
Use this to check if your environment is correctly set up before running the full FunSearch.
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model_loading(model_path, load_model=False):
    """Test if the model can be found and loaded."""
    print(f"Testing model loading from: {model_path}")
    
    # Check if path exists
    if not os.path.exists(model_path):
        print(f"ERROR: Path {model_path} does not exist")
        return False
    
    # Get directory info
    if os.path.isdir(model_path):
        print(f"Path is a directory")
        dir_contents = os.listdir(model_path)
        print(f"Directory contains {len(dir_contents)} files/folders")
        print(f"Sample contents: {dir_contents[:5]}")
    else:
        print(f"Path is a file, checking parent directory")
        parent_dir = os.path.dirname(model_path)
        if os.path.isdir(parent_dir):
            print(f"Parent directory: {parent_dir}")
            dir_contents = os.listdir(parent_dir)
            print(f"Parent directory contains {len(dir_contents)} files/folders")
            print(f"Sample contents: {dir_contents[:5]}")
    
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
        sys.exit(1)

if __name__ == "__main__":
    main() 