#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

def download_file(url, output_path):
    """Download a file using curl or wget"""
    if os.system("which curl > /dev/null") == 0:
        cmd = ["curl", "-L", url, "-o", output_path, "--progress-bar"]
    elif os.system("which wget > /dev/null") == 0:
        cmd = ["wget", url, "-O", output_path, "--show-progress"]
    else:
        print("Error: Neither curl nor wget is available")
        return False
    
    print(f"Downloading: {url} to {output_path}")
    result = subprocess.run(cmd)
    return result.returncode == 0

def download_llama3_2():
    """Download Llama 3.2 8B model from available sources"""
    
    # Create directory structure
    base_dir = Path("models/llama-3.2-8b-instruct")
    os.makedirs(base_dir, exist_ok=True)
    
    print("Downloading Llama 3.2 8B Instruct model...")
    
    # URLs for downloading Llama 3.2 directly from Meta's public repository
    # These URLs may change or be removed; this is for educational purposes
    base_url = "https://huggingface.co/meta-llama/Llama-3.2-8B-Instruct/resolve/main"
    
    files_to_download = [
        # Model configuration
        "config.json",
        # Tokenizer files
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        # Model weights (in safetensors format for easier access)
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        # Generation configuration
        "generation_config.json",
        # Special tokens map
        "special_tokens_map.json"
    ]
    
    # Download each file
    success = True
    for file in files_to_download:
        url = f"{base_url}/{file}"
        output_path = base_dir / file
        
        if output_path.exists():
            print(f"File already exists: {output_path}")
            continue
            
        if not download_file(url, output_path):
            print(f"Failed to download {file}")
            success = False
            break
    
    if success:
        print("\nLlama 3.2 8B model downloaded successfully!")
        print(f"Model stored in: {os.path.abspath(base_dir)}")
    else:
        print("\nFailed to download Llama 3.2 8B model.")
        print("Note: Meta's models require authentication on Hugging Face.")
        print("Alternative download options:")
        print("1. Try downloading from Meta's website directly: https://llama.meta.com/")
        print("2. Use a community mirror (search on GitHub for 'Llama 3.2 weights mirror')")
        print("3. Set up proper authentication with Hugging Face")

if __name__ == "__main__":
    download_llama3_2() 