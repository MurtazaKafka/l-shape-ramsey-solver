#!/usr/bin/env python3
"""
Download a verified working Llama model from Hugging Face and update llama_funsearch.py
"""
import os
import sys
import shutil
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download

def download_model(repo_id, filename, output_path):
    """Download a model file from Hugging Face."""
    print(f"Downloading {filename} from {repo_id}...")
    
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use huggingface_hub to download
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=os.path.dirname(output_path),
        )
        
        # Move to final destination if needed
        if file_path != output_path:
            if os.path.exists(output_path):
                os.remove(output_path)
            shutil.move(file_path, output_path)
        
        print(f"✅ Successfully downloaded model to: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading from Hugging Face: {e}")
        return False

def update_script(model_path):
    """Update the llama_funsearch.py script to use the downloaded model."""
    script_path = "llama_funsearch.py"
    
    # Create a backup of the original if it doesn't exist
    backup_path = "llama_funsearch.py.original_backup"
    if not os.path.exists(backup_path):
        shutil.copy2(script_path, backup_path)
        print(f"✅ Created backup of original script at: {backup_path}")
    
    with open(script_path, "r") as f:
        content = f.read()
    
    # Update the model path in the _load_model function
    updated = False
    
    if "primary_model =" in content:
        import re
        # Replace the primary model path with our new one
        content = re.sub(
            r"primary_model\s*=\s*['\"].*['\"]",
            f"primary_model = '{model_path}'",
            content
        )
        updated = True
    
    if updated:
        with open(script_path, "w") as f:
            f.write(content)
        print(f"✅ Updated {script_path} to use model: {model_path}")
        return True
    else:
        print(f"❌ Couldn't update model path in {script_path}")
        return False

def main():
    # Try downloading different models in order of preference
    models = [
        {
            "repo_id": "TheBloke/Llama-3-8B-Instruct-GGUF",
            "filename": "llama-3-8b-instruct.Q4_K_M.gguf",
            "output_path": "./models/Llama-3-8B-Instruct-Working.Q4_K_M.gguf"
        },
        {
            "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "output_path": "./models/TinyLlama-1.1B-Working.Q4_K_M.gguf"
        },
        {
            "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
            "filename": "llama-2-7b-chat.Q4_K_M.gguf",
            "output_path": "./models/Llama-2-7B-Working.Q4_K_M.gguf"
        }
    ]
    
    success = False
    downloaded_model = None
    
    for model in models:
        print(f"\n--- Trying {model['repo_id']} ---")
        if download_model(model['repo_id'], model['filename'], model['output_path']):
            downloaded_model = model['output_path']
            success = True
            break
    
    if success:
        update_script(downloaded_model)
        print(f"\n✅ Ready to run with model: {downloaded_model}")
        print("Run: python llama_funsearch.py --grid-size 3 --iterations 2")
    else:
        print("\n❌ Failed to download any working model")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())