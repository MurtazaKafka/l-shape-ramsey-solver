#!/usr/bin/env python3
"""
Simplified script to convert Meta Llama 3.3 70B model to GGUF format using llama.cpp
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

def run_cmd(cmd, cwd=None):
    """Run a command and print output"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, text=True)
    return result.returncode == 0

def main():
    # Set up paths
    meta_model_path = os.path.expanduser("~/.llama/checkpoints/Llama3.3-70B-Instruct")
    gguf_output_path = "./models/Meta-Llama-3.3-70B-Instruct.Q4_K_M.gguf"
    
    print(f"Meta model path: {meta_model_path}")
    print(f"GGUF output path: {gguf_output_path}")
    
    # Clone llama.cpp if it doesn't exist
    if not os.path.exists("llama.cpp"):
        if not run_cmd("git clone https://github.com/ggerganov/llama.cpp.git --depth 1"):
            print("Failed to clone llama.cpp repository")
            return 1
    
    # Build llama.cpp using CMake (as Makefile is deprecated)
    os.chdir("llama.cpp")
    if not os.path.exists("build"):
        os.makedirs("build", exist_ok=True)
    
    # Configure and build
    if not run_cmd("cmake -B build", cwd="./"):
        print("CMake configuration failed")
        return 1
    
    if not run_cmd("cmake --build build --config Release", cwd="./"):
        print("CMake build failed")
        return 1
    
    # Convert to GGUF format using convert.py
    print("\nAttempting GGUF conversion...")
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(f"../{gguf_output_path}"), exist_ok=True)
    
    # First try the normal converter
    conversion_cmd = f"python3 convert.py --outtype q4_k_m --outfile ../{gguf_output_path} {meta_model_path}"
    if run_cmd(conversion_cmd):
        print(f"✅ Successfully converted to GGUF format at: {gguf_output_path}")
    else:
        # Try alternative converter for Meta format
        print("First conversion attempt failed, trying with Meta format...")
        conversion_cmd = f"python3 --meta-llama --outtype q4_k_m --outfile ../{gguf_output_path} {meta_model_path}"
        if run_cmd(conversion_cmd):
            print(f"✅ Successfully converted to GGUF format at: {gguf_output_path}")
        else:
            print("❌ Failed to convert to GGUF format")
            return 1
    
    # Return to original directory and update llama_funsearch.py
    os.chdir("..")
    
    # Update llama_funsearch.py to use the new model
    script_path = "llama_funsearch.py"
    backup_path = "llama_funsearch.py.70b_backup"
    
    # Create a backup if it doesn't exist
    if not os.path.exists(backup_path):
        shutil.copy2(script_path, backup_path)
    
    # Update the model path in the file
    with open(script_path, "r") as f:
        content = f.read()
    
    content = content.replace(
        "primary_model = './models/Llama-3-8B-Working.Q4_K_M.gguf'",
        f"primary_model = '{gguf_output_path}'"
    )
    
    with open(script_path, "w") as f:
        f.write(content)
    
    print(f"✅ Updated {script_path} to use the 70B model at {gguf_output_path}")
    print("You can now run: python llama_funsearch.py --grid-size 3 --iterations 2")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())