#!/usr/bin/env python3
"""
Convert Meta's original Llama model format to Hugging Face format.
This script handles installation of required dependencies.
"""

import os
import sys
import subprocess
import argparse
import glob
import json
from pathlib import Path

def check_and_install_dependencies():
    """Check and install required dependencies."""
    dependencies = [
        "torch",
        "transformers",
        "sentencepiece",
        "blobfile",
        "protobuf",
        "accelerate"
    ]
    
    print("Checking and installing dependencies...")
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} is already installed")
        except ImportError:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} installed successfully")

def download_conversion_script():
    """Download the conversion script from Hugging Face if not present."""
    target_path = "convert_llama_weights_to_hf.py"
    
    if os.path.exists(target_path):
        print(f"Conversion script already exists at {target_path}")
        return target_path
    
    url = "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/convert_llama_weights_to_hf.py"
    
    print(f"Downloading conversion script from {url}...")
    try:
        import requests
        response = requests.get(url)
        response.raise_for_status()
        
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        
        print(f"Conversion script downloaded to {target_path}")
        return target_path
    except Exception as e:
        print(f"Error downloading conversion script: {e}")
        print("Trying with subprocess...")
        
        try:
            subprocess.check_call(["wget", url, "-O", target_path])
            print(f"Conversion script downloaded to {target_path}")
            return target_path
        except Exception as wget_e:
            print(f"Error downloading with wget: {wget_e}")
            print("Trying with curl...")
            
            try:
                subprocess.check_call(["curl", "-o", target_path, url])
                print(f"Conversion script downloaded to {target_path}")
                return target_path
            except Exception as curl_e:
                print(f"Error downloading with curl: {curl_e}")
                raise RuntimeError("Failed to download conversion script. Please download it manually.")

def get_model_size_from_params(input_dir):
    """Determine the model size from params.json if available."""
    params_path = os.path.join(input_dir, "params.json")
    
    if os.path.exists(params_path):
        try:
            with open(params_path, 'r') as f:
                params = json.load(f)
            
            dim = params.get("dim")
            n_layers = params.get("n_layers")
            
            if dim and n_layers:
                # Very rough estimation of parameters
                vocab_size = params.get("vocab_size", 32000)
                hidden_dim = params.get("hidden_dim", 4 * dim)
                total_params = n_layers * (12 * dim * dim + 2 * dim * hidden_dim) + vocab_size * dim
                
                total_params_b = total_params / 1e9
                
                if total_params_b >= 65:
                    return "70B"
                elif total_params_b >= 30:
                    return "35B"
                elif total_params_b >= 12:
                    return "13B"
                elif total_params_b >= 6:
                    return "7B"
                else:
                    return "7B"  # Default to 7B for smaller models
        except Exception as e:
            print(f"Error reading params.json: {e}")
    
    # Count number of consolidated.*.pth files to estimate model size
    consolidated_files = glob.glob(os.path.join(input_dir, "consolidated.*.pth"))
    if consolidated_files:
        if len(consolidated_files) >= 80:
            return "70B"
        elif len(consolidated_files) >= 32:
            return "35B"
        elif len(consolidated_files) >= 16:
            return "13B"
        else:
            return "7B"
    
    # Default to 70B if we can't determine
    print("Could not determine model size. Using 70B as default.")
    return "70B"

def convert_model(input_dir, output_dir, model_size=None):
    """Convert the model using the HF conversion script."""
    script_path = download_conversion_script()
    
    if not model_size:
        model_size = get_model_size_from_params(input_dir)
    
    # Add "Llama-3-" prefix if not already present
    if not model_size.startswith("Llama-3-"):
        model_size = f"Llama-3-{model_size}"
    
    print(f"Converting model from {input_dir} to {output_dir} with size {model_size}...")
    
    cmd = [
        sys.executable,
        script_path,
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--model_size", model_size
    ]
    
    try:
        subprocess.check_call(cmd)
        print(f"✅ Model converted successfully to {output_dir}")
        
        # Verify output
        config_file = os.path.join(output_dir, "config.json")
        if os.path.exists(config_file):
            print(f"✅ Verified config.json exists in {output_dir}")
        else:
            print(f"⚠️ config.json not found in {output_dir}. Conversion may have failed.")
        
        return True
    except Exception as e:
        print(f"Error converting model: {e}")
        return False

def validate_input_dir(input_dir):
    """Validate that the input directory contains Meta's Llama format files."""
    # Check for consolidated.*.pth files
    consolidated_files = glob.glob(os.path.join(input_dir, "consolidated.*.pth"))
    if not consolidated_files:
        print(f"⚠️ No consolidated.*.pth files found in {input_dir}")
        return False
    
    # Check for tokenizer.model
    tokenizer_file = os.path.join(input_dir, "tokenizer.model")
    if not os.path.exists(tokenizer_file):
        print(f"⚠️ tokenizer.model not found in {input_dir}")
        return False
    
    print(f"✅ Valid Meta Llama format files found in {input_dir}")
    return True

def main():
    """Main function to run the conversion process."""
    parser = argparse.ArgumentParser(description="Convert Meta's Llama model format to Hugging Face format")
    parser.add_argument("--input_dir", type=str, default="/home/DAVIDSON/munikzad/.llama/checkpoints/Llama3.3-70B-Instruct",
                        help="Input directory containing Meta's Llama model files")
    parser.add_argument("--output_dir", type=str, default="./llama3_hf",
                        help="Output directory for the converted Hugging Face model")
    parser.add_argument("--model_size", type=str, default=None,
                        help="Model size (e.g., '7B', '13B', '70B')")
    parser.add_argument("--skip_deps", action="store_true",
                        help="Skip dependency installation")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if output directory already contains a converted model
    config_file = os.path.join(args.output_dir, "config.json")
    if os.path.exists(config_file):
        print(f"⚠️ Output directory {args.output_dir} already contains a config.json file.")
        overwrite = input("Do you want to overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            print("Exiting without conversion.")
            return
    
    # Check and install dependencies
    if not args.skip_deps:
        check_and_install_dependencies()
    
    # Validate input directory
    if not validate_input_dir(args.input_dir):
        print("Input directory validation failed. Continuing anyway...")
    
    # Convert the model
    success = convert_model(args.input_dir, args.output_dir, args.model_size)
    
    if success:
        print("\n===============================================")
        print("✅ Conversion complete!")
        print("===============================================")
        print(f"The Hugging Face model is now available at: {args.output_dir}")
        print("\nTo use this model with llama_funsearch.py, update the model_path:")
        print(f"python llama_funsearch.py --model_path {args.output_dir}")
        print("===============================================")
    else:
        print("\n===============================================")
        print("❌ Conversion failed.")
        print("Please check the error messages above.")
        print("===============================================")

if __name__ == "__main__":
    main() 