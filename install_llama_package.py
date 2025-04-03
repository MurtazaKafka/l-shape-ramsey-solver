#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
import os

def install_llama_cpp():
    """Install the llama-cpp-python package."""
    print("Installing llama-cpp-python...")
    try:
        # Install with pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
        print("Successfully installed llama-cpp-python")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing llama-cpp-python: {e}")
        return False

def verify_model_path():
    """Verify that the Llama 3.2 model exists."""
    model_path = Path.home() / ".llama" / "checkpoints" / "Llama3.2-3B-Instruct"
    
    if model_path.exists():
        print(f"Llama 3.2 model found at: {model_path}")
        
        # Check for essential files
        required_files = ["consolidated.00.pth", "params.json", "tokenizer.model"]
        missing_files = [f for f in required_files if not (model_path / f).exists()]
        
        if missing_files:
            print(f"Warning: Missing required files: {', '.join(missing_files)}")
            return False
        else:
            print("All required model files found")
            return True
    else:
        print(f"Error: Llama 3.2 model not found at {model_path}")
        return False

def setup_llama_cpp_for_meta_model():
    """Setup llama-cpp-python for use with Meta's Llama model."""
    print("Creating a simple conversion script for Meta's Llama model format...")
    
    # Path to the script
    script_path = Path("convert_meta_to_gguf.py")
    
    # Write the conversion script
    script_content = """
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
import json
import os
import shutil

def convert_meta_to_gguf(model_path):
    \"\"\"Convert Meta's Llama model format to GGUF format for llama-cpp-python.\"\"\"
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist.")
        return False
    
    # Create output directory
    output_dir = model_path.with_name(f"{model_path.name}-gguf")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for required files
    required_files = ["consolidated.00.pth", "params.json", "tokenizer.model"]
    for req_file in required_files:
        if not (model_path / req_file).exists():
            print(f"Error: Required file {req_file} not found in {model_path}")
            return False
    
    # Load model parameters
    with open(model_path / "params.json", "r") as f:
        params = json.load(f)
    
    # Create GGUF-compatible config
    config = {
        "vocab_size": params.get("vocab_size", 32000),
        "hidden_size": params.get("dim", 4096),
        "intermediate_size": params.get("ffn_dim", 11008),
        "num_hidden_layers": params.get("n_layers", 32),
        "num_attention_heads": params.get("n_heads", 32),
        "hidden_act": "silu",
        "max_position_embeddings": params.get("max_seq_len", 2048),
        "tokenizer_model": str(model_path / "tokenizer.model"),
    }
    
    # Write config.json
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Copy tokenizer model
    shutil.copy(model_path / "tokenizer.model", output_dir / "tokenizer.model")
    
    # Copy README with instructions
    with open(output_dir / "README.md", "w") as f:
        f.write(f"# Converted Llama 3.2 Model for llama-cpp-python\\n\\n")
        f.write(f"Original model: {model_path}\\n\\n")
        f.write("To use this model with llama-cpp-python, run:\\n")
        f.write("```python\\n")
        f.write("from llama_cpp import Llama\\n\\n")
        f.write(f"model = Llama(model_path='{output_dir / 'model.gguf'}')\\n")
        f.write("```\\n")
    
    # Note: Actual conversion of weights would require more complex code
    # This is just a placeholder to demonstrate the process
    with open(output_dir / "model.gguf", "w") as f:
        f.write("This is a placeholder for the converted model.\\n")
        f.write("Actual conversion requires more complex processing.\\n")
    
    print(f"Model format prepared at {output_dir}")
    print("Note: This is a simplified preparation and doesn't perform the actual weight conversion.")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_meta_to_gguf.py <path_to_meta_llama_model>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    success = convert_meta_to_gguf(model_path)
    
    if success:
        print("Conversion preparation completed. Please use llama-cpp-python with the appropriate model format.")
    else:
        print("Conversion preparation failed.")
        sys.exit(1)
"""
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    print(f"Created conversion script at {script_path}")
    print("Note: Full conversion of Meta's model format to GGUF format requires additional steps.")
    print("This script serves as a starting point for working with llama-cpp-python.")

def main():
    """Main function to install and verify llama-cpp-python."""
    # Install llama-cpp-python
    if not install_llama_cpp():
        print("Failed to install llama-cpp-python. Please try manually.")
        return
    
    # Verify model path
    if not verify_model_path():
        print("Model verification failed.")
        return
    
    # Setup llama-cpp-python for Meta's model format
    setup_llama_cpp_for_meta_model()
    
    print("\nNext steps:")
    print("1. If needed, complete the conversion of Meta's model format to GGUF format")
    print("2. Update llama3_solver.py to use the converted model")
    print("3. Run the solver with: python llama3_solver.py")

if __name__ == "__main__":
    main() 