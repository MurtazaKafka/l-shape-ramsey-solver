#!/usr/bin/env python3
"""
Download Llama 3.3 70B GGUF model file from Hugging Face.
This script downloads the GGUF file for the Llama 3.3 70B model.
"""

import os
import sys
from huggingface_hub import hf_hub_download

def download_llama_70b_gguf():
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    repo_id = "TheBloke/Llama-3-70B-Instruct-GGUF"
    filename = "llama-3-70b-instruct.Q4_K_M.gguf"
    local_path = os.path.join(model_dir, filename)
    print(f"Starting download of {filename} from {repo_id}...")
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        if file_path != local_path:
            os.rename(file_path, local_path)
        print(f"Model downloaded successfully to {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

def main():
    path = download_llama_70b_gguf()
    if path:
        print(f"Model ready at: {path}")
    else:
        print("Failed to download Llama 3.3 70B GGUF model.")
        sys.exit(1)

if __name__ == "__main__":
    main()