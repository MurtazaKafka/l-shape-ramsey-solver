#!/usr/bin/env python3
"""
Download the Llama 3.3 70B GGUF model from Hugging Face.
"""
import os
import sys
import shutil
import requests
from huggingface_hub import hf_hub_download
from tqdm import tqdm

def download_with_progress(url, output_path):
    """Download a file with progress tracking"""
    print(f"Downloading from: {url}")
    print(f"Output path: {output_path}")
    
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1 MB
    
    with open(output_path, 'wb') as f, tqdm(
            total=total_size, 
            unit='B',
            unit_scale=True,
            desc=output_path.split('/')[-1]) as bar:
        for data in response.iter_content(block_size):
            f.write(data)
            bar.update(len(data))
    
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0

def download_from_huggingface():
    """Download the model using the huggingface_hub library"""
    model_dir = "./models"
    model_output = os.path.join(model_dir, "Llama-3.3-70B-Instruct-Q4_K_M.gguf")
    
    try:
        print("Attempting to download using huggingface_hub...")
        path = hf_hub_download(
            repo_id="bartowski/Llama-3.3-70B-Instruct-GGUF",
            filename="Llama-3.3-70B-Instruct-Q4_K_M.gguf",
            local_dir=model_dir
        )
        if os.path.exists(path):
            print(f"✅ Successfully downloaded model to: {path}")
            return path
    except Exception as e:
        print(f"❌ Error with huggingface_hub download: {e}")
    
    return None

def download_with_wget():
    """Download using wget command"""
    model_dir = "./models"
    model_output = os.path.join(model_dir, "Llama-3.3-70B-Instruct-Q4_K_M.gguf")
    
    # URL to the raw file, not the blob view
    url = "https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF/resolve/main/Llama-3.3-70B-Instruct-Q4_K_M.gguf"
    
    os.makedirs(model_dir, exist_ok=True)
    
    print("Attempting to download using wget...")
    cmd = f"wget -q --show-progress {url} -O {model_output}"
    print(f"Running: {cmd}")
    
    ret = os.system(cmd)
    if ret == 0 and os.path.exists(model_output):
        print(f"✅ Successfully downloaded model to: {model_output}")
        return model_output
    else:
        print("❌ wget download failed")
        return None

def download_with_requests():
    """Download using requests library"""
    model_dir = "./models"
    model_output = os.path.join(model_dir, "Llama-3.3-70B-Instruct-Q4_K_M.gguf")
    
    # URL to the raw file, not the blob view
    url = "https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF/resolve/main/Llama-3.3-70B-Instruct-Q4_K_M.gguf"
    
    print("Attempting to download using requests...")
    result = download_with_progress(url, model_output)
    
    if result:
        print(f"✅ Successfully downloaded model to: {model_output}")
        return model_output
    else:
        print("❌ requests download failed")
        return None

def update_funsearch_script(model_path):
    """Update the llama_funsearch.py script to use the downloaded model"""
    script_path = "llama_funsearch.py"
    backup_path = "llama_funsearch.py.70b_backup"
    
    # Create backup if it doesn't exist
    if not os.path.exists(backup_path):
        shutil.copy2(script_path, backup_path)
    
    # Update the script to use the new model
    with open(script_path, "r") as f:
        content = f.read()
    
    # Replace the primary model path
    import re
    content = re.sub(
        r"primary_model\s*=\s*['\"].*['\"]",
        f"primary_model = '{model_path}'",
        content
    )
    
    with open(script_path, "w") as f:
        f.write(content)
    
    print(f"✅ Updated {script_path} to use model: {model_path}")

def main():
    print("=" * 70)
    print("Downloading Llama 3.3 70B GGUF Model")
    print("=" * 70)
    
    # Try different download methods
    download_path = download_from_huggingface()
    
    if not download_path:
        download_path = download_with_wget()
    
    if not download_path:
        download_path = download_with_requests()
    
    if download_path:
        update_funsearch_script(download_path)
        
        print("\n" + "=" * 70)
        print(f"✅ Model ready at: {download_path}")
        print("You can now run: python llama_funsearch.py --grid-size 3 --iterations 2")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("❌ Failed to download the model. Please check the URL or your internet connection.")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())