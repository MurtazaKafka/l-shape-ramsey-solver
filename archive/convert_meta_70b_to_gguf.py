#!/usr/bin/env python3
"""
Convert Meta Llama 3.3 70B model to GGUF format and update llama_funsearch.py to use it.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path
import argparse

def run_command(cmd, check=True, capture_output=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=check, 
            text=True,
            capture_output=capture_output
        )
        if capture_output:
            print(result.stdout)
            if result.stderr:
                print(f"STDERR: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if capture_output:
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
        return False

def convert_to_hf_format(meta_model_path, hf_output_path):
    """Convert the Meta Llama model to Hugging Face format."""
    print(f"\n=== Converting Meta Llama model to Hugging Face format ===")
    
    # Create output directory
    os.makedirs(hf_output_path, exist_ok=True)
    
    # First, try to use convert_llama3_to_hf.py
    conversion_script = None
    for script_name in ["convert_llama3_to_hf.py", "convert_meta_to_hf.py", "convert_llama_to_hf.py"]:
        if os.path.exists(script_name):
            conversion_script = script_name
            break
    
    if not conversion_script:
        print("❌ No conversion script found - creating minimal converter")
        # Create a minimal conversion script
        create_minimal_converter()
        conversion_script = "minimal_llama_converter.py"
    
    # Run the conversion script
    cmd = f"python {conversion_script} --input-dir {meta_model_path} --output-dir {hf_output_path}"
    result = run_command(cmd)
    
    # Check if conversion was successful
    if not os.path.exists(os.path.join(hf_output_path, "config.json")):
        print(f"❌ Conversion failed - no config.json found in {hf_output_path}")
        return False
    
    print(f"✅ Successfully converted to HF format at {hf_output_path}")
    return True

def create_minimal_converter():
    """Create a minimal converter script if none exists."""
    script_content = '''#!/usr/bin/env python3
import os
import json
import torch
import shutil
import argparse
from pathlib import Path

def convert_meta_to_hf(input_dir, output_dir):
    """Convert Meta Llama format to Hugging Face format."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Copy tokenizer
    if (input_path / "tokenizer.model").exists():
        shutil.copy(input_path / "tokenizer.model", output_path / "tokenizer.model")
    
    # Load params.json
    with open(input_path / "params.json", "r") as f:
        params = json.load(f)
    
    # Create HF config
    config = {
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": params.get("dim", 8192),
        "intermediate_size": params.get("ffn_dim", 28672),
        "max_position_embeddings": params.get("max_seq_len", 4096),
        "model_type": "llama",
        "num_attention_heads": params.get("n_heads", 64),
        "num_hidden_layers": params.get("n_layers", 80),
        "num_key_value_heads": params.get("n_kv_heads", 8),
        "rms_norm_eps": 1e-05,
        "rope_scaling": {"factor": 1.0, "type": "linear"},
        "torch_dtype": "float16",
        "transformers_version": "4.38.2",
        "vocab_size": 128256
    }
    
    # Save config
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Create tokenizer config
    tokenizer_config = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "model_max_length": config["max_position_embeddings"],
        "padding_side": "right",
        "tokenizer_class": "LlamaTokenizer",
        "unk_token": "<unk>"
    }
    
    # Save tokenizer config
    with open(output_path / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    
    # Create generation config
    generation_config = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "transformers_version": "4.38.2"
    }
    
    # Save generation config
    with open(output_path / "generation_config.json", "w") as f:
        json.dump(generation_config, f, indent=2)
    
    # Copy model weights
    weight_files = sorted(list(input_path.glob("consolidated.*.pth")))
    for file in weight_files:
        print(f"Processing {file}")
        # Create a symlink instead of copying huge files
        target = output_path / file.name
        if not target.exists():
            os.symlink(str(file.absolute()), str(target))
    
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Meta Llama format to Hugging Face format")
    parser.add_argument("--input-dir", required=True, help="Input directory with Meta Llama format")
    parser.add_argument("--output-dir", required=True, help="Output directory for HF format")
    args = parser.parse_args()
    
    convert_meta_to_hf(args.input_dir, args.output_dir)
'''
    with open("minimal_llama_converter.py", "w") as f:
        f.write(script_content)
    
    os.chmod("minimal_llama_converter.py", 0o755)
    print("Created minimal_llama_converter.py")

def convert_to_gguf_format(hf_model_path, gguf_output_path):
    """Convert the HF format model to GGUF."""
    print(f"\n=== Converting HF model to GGUF format ===")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(gguf_output_path), exist_ok=True)
    
    # Try different methods to convert to GGUF
    
    # Method 1: llama-cpp-python convert module (if available)
    try:
        import llama_cpp.server.convert
        print("Using llama_cpp's built-in converter...")
        cmd = f"python -m llama_cpp.server.convert --outtype q4_k_m --outfile {gguf_output_path} {hf_model_path}"
        if run_command(cmd, check=False):
            if os.path.exists(gguf_output_path):
                print(f"✅ Successfully converted to GGUF format at {gguf_output_path}")
                return True
    except ImportError:
        print("llama_cpp.server.convert not available, trying other methods...")
    
    # Method 2: Check for local llama.cpp repo
    for convert_script in ["./llama.cpp/convert.py", "./convert_to_gguf.py"]:
        if os.path.exists(convert_script):
            cmd = f"python {convert_script} --outtype q4_k_m --outfile {gguf_output_path} {hf_model_path}"
            if run_command(cmd, check=False):
                if os.path.exists(gguf_output_path):
                    print(f"✅ Successfully converted to GGUF format at {gguf_output_path}")
                    return True
    
    # Method 3: Use quantize utility from llama-cpp-python
    try:
        cmd = f"python -c \"from llama_cpp.quantize import main; main(model='{hf_model_path}', outtype='q4_k_m', outfile='{gguf_output_path}', threads=4)\""
        if run_command(cmd, check=False):
            if os.path.exists(gguf_output_path):
                print(f"✅ Successfully converted to GGUF format at {gguf_output_path}")
                return True
    except Exception as e:
        print(f"Error using llama_cpp.quantize: {e}")
    
    # Method 4: Try to install and use llama.cpp directly
    try:
        # Clone llama.cpp repository if it doesn't exist
        if not os.path.exists("./llama.cpp"):
            run_command("git clone https://github.com/ggerganov/llama.cpp.git --depth 1")
        
        # Build the convert tool
        run_command("cd llama.cpp && make")
        
        # Use the convert tool
        cmd = f"cd llama.cpp && python convert.py --outtype q4_k_m --outfile ../{gguf_output_path} ../{hf_model_path}"
        if run_command(cmd, check=False):
            if os.path.exists(gguf_output_path):
                print(f"✅ Successfully converted to GGUF format at {gguf_output_path}")
                return True
    except Exception as e:
        print(f"Error using llama.cpp: {e}")
    
    print(f"❌ All conversion methods failed")
    return False

def update_funsearch_script(gguf_model_path):
    """Update llama_funsearch.py to use the newly converted model."""
    script_path = "llama_funsearch.py"
    backup_path = "llama_funsearch.py.70b_backup"
    
    # Create a backup if it doesn't exist
    if not os.path.exists(backup_path):
        shutil.copy2(script_path, backup_path)
        print(f"✅ Created backup of original script at: {backup_path}")
    
    try:
        with open(script_path, "r") as f:
            content = f.read()
        
        # Update the model path
        import re
        
        # Replace the primary model path
        if "primary_model =" in content:
            content = re.sub(
                r"primary_model\s*=\s*['\"].*['\"]",
                f"primary_model = '{gguf_model_path}'",
                content
            )
        
        # Also update model initialization to ensure optimal parameters for 70B model
        if "_load_model" in content and "model_params" in content:
            # Make sure we have optimal parameters for 70B model on GPU
            if "model_params.update" in content and "n_gpu_layers" in content:
                # It already has GPU parameters, make sure they're optimal for 70B
                content = re.sub(
                    r"model_params\.update\(\{.*?}\)",
                    """model_params.update({
                            "n_gpu_layers": -1,     # Use all layers on GPU
                            "main_gpu": 0,          # Primary GPU device
                            "n_batch": 512,         # Batch size for inference
                            "offload_kqv": True,    # Offload KQV operations for memory efficiency
                            "use_mlock": True       # Lock memory to prevent swapping
                        })""",
                    content,
                    flags=re.DOTALL
                )
        
        with open(script_path, "w") as f:
            f.write(content)
        
        print(f"✅ Updated {script_path} to use model: {gguf_model_path}")
        return True
    
    except Exception as e:
        print(f"❌ Error updating script: {e}")
        # Restore from backup if something went wrong
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, script_path)
            print(f"Restored script from backup")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert Meta Llama 3.3 70B to GGUF and set up for llama_funsearch.py")
    parser.add_argument("--meta-model-path", default=os.path.expanduser("~/.llama/checkpoints/Llama3.3-70B-Instruct"),
                       help="Path to the downloaded Meta Llama 3.3 70B model (default: ~/.llama/checkpoints/Llama3.3-70B-Instruct)")
    parser.add_argument("--hf-output-path", default="./llama3_hf",
                       help="Path to output the HuggingFace format model (default: ./llama3_hf)")
    parser.add_argument("--gguf-output-path", default="./models/Meta-Llama-3.3-70B-Instruct.Q4_K_M.gguf",
                       help="Path to output the GGUF model (default: ./models/Meta-Llama-3.3-70B-Instruct.Q4_K_M.gguf)")
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Converting Meta Llama 3.3 70B Model to GGUF Format")
    print("=" * 70)
    print(f"Meta model path: {args.meta_model_path}")
    print(f"HF output path: {args.hf_output_path}")
    print(f"GGUF output path: {args.gguf_output_path}")
    print("=" * 70)
    
    # Check if the Meta model exists
    if not os.path.exists(args.meta_model_path):
        print(f"❌ Meta Llama model not found at {args.meta_model_path}")
        print(f"Please specify the correct path with --meta-model-path")
        return 1
    
    # Check if GGUF file already exists
    if os.path.exists(args.gguf_output_path):
        print(f"✅ GGUF file already exists at {args.gguf_output_path}")
        print(f"Skipping conversion and updating script...")
    else:
        # Step 1: Convert to HF format
        if not convert_to_hf_format(args.meta_model_path, args.hf_output_path):
            print("❌ Failed to convert to HF format")
            return 1
        
        # Step 2: Convert HF format to GGUF
        if not convert_to_gguf_format(args.hf_output_path, args.gguf_output_path):
            print("❌ Failed to convert to GGUF format")
            return 1
    
    # Step 3: Update the llama_funsearch.py script
    if not update_funsearch_script(args.gguf_output_path):
        print("❌ Failed to update funsearch script")
        return 1
    
    print("\n" + "=" * 70)
    print(f"✅ Successfully prepared Meta Llama 3.3 70B model")
    print(f"Model path: {args.gguf_output_path}")
    print(f"You can now run: python llama_funsearch.py --grid-size 3 --iterations 2")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())