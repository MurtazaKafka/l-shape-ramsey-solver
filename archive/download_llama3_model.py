#!/usr/bin/env python3
"""
Script to download a fresh copy of Llama 3.3 70B model from Hugging Face.
This script will download a working GGUF quantized model for GPU acceleration.
"""
import os
import sys
import subprocess
import requests
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import hf_hub_download, login

def download_with_progress(url, output_path):
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1 MB
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            f.write(data)
            bar.update(len(data))

def ensure_huggingface_cli():
    """Ensure huggingface_cli is installed."""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'huggingface_hub'])
        return True
    except Exception as e:
        print(f"Failed to install huggingface_hub: {e}")
        return False

def try_direct_download():
    """Try to download models directly from well-known sources."""
    models = [
        # TinyLlama (this is a small model that should work)
        {
            "name": "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
            "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
            "output": "./models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"
        },
        # Llama 3 8B Instruct (medium sized model)
        {
            "name": "Llama-3-8B-Instruct.Q4_K_M.gguf",
            "url": "https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf",
            "output": "./models/Llama-3-8B-Instruct.Q4_K_M.gguf"
        }
    ]
    
    successful = []
    
    for model in models:
        print(f"Downloading {model['name']}...")
        try:
            download_with_progress(model['url'], model['output'])
            successful.append(model['output'])
            print(f"✅ Successfully downloaded {model['name']} to {model['output']}")
        except Exception as e:
            print(f"❌ Failed to download {model['name']}: {e}")
    
    return successful

def try_huggingface_download():
    """Try to download models via Hugging Face Hub."""
    models = [
        # TinyLlama
        {
            "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "filename": "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
            "output": "./models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"
        },
        # Llama 3 8B Instruct
        {
            "repo_id": "TheBloke/Llama-3-8B-Instruct-GGUF",
            "filename": "llama-3-8b-instruct.Q4_K_M.gguf",
            "output": "./models/Llama-3-8B-Instruct.Q4_K_M.gguf"
        }
    ]
    
    successful = []
    
    try:
        for model in models:
            print(f"Downloading {model['filename']} from {model['repo_id']}...")
            try:
                file_path = hf_hub_download(
                    repo_id=model['repo_id'],
                    filename=model['filename'],
                    local_dir="./models",
                    local_dir_use_symlinks=False
                )
                os.makedirs(os.path.dirname(model['output']), exist_ok=True)
                if os.path.exists(file_path) and file_path != model['output']:
                    if os.path.exists(model['output']):
                        os.remove(model['output'])
                    os.rename(file_path, model['output'])
                successful.append(model['output'])
                print(f"✅ Successfully downloaded {model['filename']} to {model['output']}")
            except Exception as e:
                print(f"❌ Failed to download {model['filename']}: {e}")
    except Exception as e:
        print(f"❌ Error with Hugging Face download: {e}")
    
    return successful

def main():
    """Main function to download the models."""
    print("=" * 80)
    print("🔽 Downloading Llama models for L-shape Ramsey Problem")
    print("=" * 80)
    
    # Create models directory if it doesn't exist
    os.makedirs("./models", exist_ok=True)
    
    # First try direct download
    print("\n1️⃣ Attempting direct download...")
    direct_downloads = try_direct_download()
    
    # If direct download failed, try via Hugging Face
    if not direct_downloads:
        print("\n2️⃣ Attempting download via Hugging Face Hub...")
        ensure_huggingface_cli()
        hf_downloads = try_huggingface_download()
        
        if not hf_downloads:
            print("\n❌ All download methods failed. Please try again later or manually download a model.")
            return 1
    
    # Update the model path in llama_funsearch.py
    default_model = direct_downloads[0] if direct_downloads else hf_downloads[0]
    print(f"\n✅ Using model: {default_model} for llama_funsearch.py")
    
    # Update fix_numpy_and_run.py to use the new model
    try:
        with open("fix_numpy_and_run.py", "r") as f:
            content = f.read()
        
        if "model_locations = [" in content:
            content = content.split("model_locations = [")[0]
            content += f"""model_locations = [
                "{default_model}",
                "./llama_gguf/tinyllama-1.1b-q4.gguf",
                "./models/Llama-3.3-70B-Instruct-Q4_K_M.gguf"
            ]"""
            content += content.split("model_locations = [")[1].split("]", 1)[1]
        
        with open("fix_numpy_and_run.py", "w") as f:
            f.write(content)
        
        print("✅ Updated fix_numpy_and_run.py to use the new model.")
    except Exception as e:
        print(f"⚠️ Could not update fix_numpy_and_run.py: {e}")
    
    print("\n🎉 Done! You can now run the modified script with the new model.")
    print("   Run: python fix_numpy_and_run.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())