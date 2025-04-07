#!/usr/bin/env python3
"""
Convert Meta's original Llama format (consolidated.*.pth) to Hugging Face format.
This script is required to convert the original Llama checkpoints to a format
that the transformers library can use.
"""

import os
import sys
import argparse
import shutil
import torch
from pathlib import Path

def check_requirements():
    """Check if the required libraries are installed."""
    try:
        import transformers
        from transformers.models.llama.convert_llama_weights_to_hf import convert_llama_weights_to_hf
        print("✓ Found transformers with Llama conversion utilities")
        return True
    except ImportError:
        print("✗ transformers library not found.")
        print("Installing transformers...")
        os.system("pip install transformers")
        try:
            import transformers
            from transformers.models.llama.convert_llama_weights_to_hf import convert_llama_weights_to_hf
            print("✓ Successfully installed transformers")
            return True
        except ImportError:
            print("✗ Failed to install transformers. Please install it manually:")
            print("pip install transformers")
            return False
    except AttributeError:
        print("✗ transformers library is installed but doesn't have the Llama conversion tool.")
        print("This might be due to an outdated version. Updating transformers...")
        os.system("pip install -U transformers")
        try:
            from transformers.models.llama.convert_llama_weights_to_hf import convert_llama_weights_to_hf
            print("✓ Successfully updated transformers")
            return True
        except AttributeError:
            print("✗ Still couldn't find the conversion tool.")
            print("Please check the transformers version:")
            os.system("pip show transformers")
            print("\nYou may need to manually download the conversion script:")
            print("wget https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/convert_llama_weights_to_hf.py")
            return False

def convert_llama(input_dir, output_dir, model_name=None):
    """Convert Llama weights to Hugging Face format."""
    print(f"Converting Llama weights from {input_dir} to {output_dir}")
    
    # Import the convert_llama_weights_to_hf function from transformers
    try:
        from transformers.models.llama.convert_llama_weights_to_hf import convert_llama_weights_to_hf
    except (ImportError, AttributeError):
        print("✗ Could not import conversion function.")
        print("Attempting to use custom implementation...")
        
        # Check if we can download the script
        try:
            import requests
            url = "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/convert_llama_weights_to_hf.py"
            response = requests.get(url)
            if response.status_code == 200:
                with open("convert_llama_weights_to_hf.py", "w") as f:
                    f.write(response.text)
                print("✓ Downloaded conversion script")
                
                # Now import from the downloaded script
                import importlib.util
                spec = importlib.util.spec_from_file_location("convert_llama_weights_to_hf", "convert_llama_weights_to_hf.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                convert_llama_weights_to_hf = module.convert_llama_weights_to_hf
            else:
                print(f"✗ Failed to download script: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Error downloading script: {e}")
            return False
    
    # Define model size based on parameter count
    if model_name is None:
        # Try to detect model size from params.json
        params_file = os.path.join(input_dir, "params.json")
        if os.path.exists(params_file):
            import json
            with open(params_file, "r") as f:
                params = json.load(f)
            
            # Calculate number of parameters (rough estimate)
            dim = params.get("dim", 0)
            n_layers = params.get("n_layers", 0)
            n_heads = params.get("n_heads", 0)
            vocab_size = params.get("vocab_size", 0)
            
            if dim > 0 and n_layers > 0:
                # Rough estimate of parameter count in billions
                param_count = (dim * dim * n_layers * 4 + dim * vocab_size) / 1e9
                print(f"Estimated parameter count: {param_count:.1f}B")
                
                # Determine model size
                if param_count >= 60:
                    model_name = "Llama-3-70B"
                elif param_count >= 30:
                    model_name = "Llama-3-8B"
                else:
                    model_name = "Llama-3-8B"
            else:
                model_name = "Llama-3"
        else:
            model_name = "Llama-3"
    
    # Check if output directory exists
    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists.")
        overwrite = input("Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != "y":
            print("Conversion aborted.")
            return False
        shutil.rmtree(output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the conversion
    print("Starting conversion (this may take a while)...")
    try:
        # Check if we're using a custom implementation
        if 'module' in locals():
            # For custom implementation
            module.convert_llama_weights_to_hf(
                input_dir=input_dir,
                output_dir=output_dir,
                model_size=model_name
            )
        else:
            # For transformers implementation
            convert_llama_weights_to_hf(
                input_dir=input_dir,
                output_dir=output_dir,
                model_size=model_name
            )
        
        print("✓ Conversion completed successfully!")
        
        # Verify the output
        if os.path.exists(os.path.join(output_dir, "config.json")):
            print("✓ Found config.json in output directory")
            # Check for model files
            model_files = [f for f in os.listdir(output_dir) if f.startswith("pytorch_model") or f.endswith(".bin")]
            if model_files:
                print(f"✓ Found {len(model_files)} model files")
            else:
                print("✗ No model files found, conversion may have failed")
                return False
        else:
            print("✗ config.json not found in output directory, conversion may have failed")
            return False
        
        return True
    
    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert Meta's Llama format to Hugging Face format")
    parser.add_argument("--input-dir", type=str, default="/home/DAVIDSON/munikzad/.llama/checkpoints/Llama3.3-70B-Instruct",
                       help="Input directory containing the Meta Llama model files")
    parser.add_argument("--output-dir", type=str, default="./llama3_hf",
                       help="Output directory for the converted Hugging Face model")
    parser.add_argument("--model-name", type=str, default=None,
                       help="Model name/size (e.g., 'Llama-3-70B')")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Llama Model Converter: Meta format to Hugging Face format")
    print("=" * 70)
    
    # Check if requirements are met
    if not check_requirements():
        print("Aborting conversion due to missing requirements.")
        sys.exit(1)
    
    # Convert the model
    success = convert_llama(args.input_dir, args.output_dir, args.model_name)
    
    if success:
        print("\nNow you can use the converted model with the transformers library:")
        print(f"model_path = '{os.path.abspath(args.output_dir)}'")
        
        # Update the llama_funsearch.py script
        update = input("Would you like to update llama_funsearch.py to use this model? (y/n): ")
        if update.lower() == "y":
            try:
                with open("llama_funsearch.py", "r") as f:
                    content = f.read()
                
                # Update the model path
                content = content.replace(
                    "self.model_path = model_path or \"/home/DAVIDSON/munikzad/.llama/checkpoints/Llama3.3-70B-Instruct\"",
                    f"self.model_path = model_path or \"{os.path.abspath(args.output_dir)}\""
                )
                
                # Write to a new file
                with open("llama_funsearch_updated.py", "w") as f:
                    f.write(content)
                
                print("✓ Created updated file: llama_funsearch_updated.py")
                print("You can now run: python llama_funsearch_updated.py")
            except Exception as e:
                print(f"✗ Error updating script: {e}")
    else:
        print("\nConversion failed. Please try again or use a different approach.")
        sys.exit(1)

if __name__ == "__main__":
    main() 