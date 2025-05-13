#!/usr/bin/env python3
"""
Convert downloaded Llama 3.3 70B model to GGUF format and prepare it for use with llama_funsearch.py.
This script converts Meta's official Llama model to GGUF format and sets it up for the L-shape Ramsey problem.
"""
import os
import sys
import shutil
import subprocess
import requests
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_with_progress(url, output_path, chunk_size=1024*1024):
    """Download a file with a progress bar."""
    print(f"Downloading {url.split('/')[-1]} to {output_path}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    
    return output_path

def convert_meta_llama_to_gguf():
    """Convert the official Meta Llama 3.3 model to GGUF format."""
    print("\n" + "=" * 70)
    print("Converting Meta's Llama 3.3 70B model to GGUF format")
    print("=" * 70)
    
    # Check if the Meta Llama model exists
    meta_llama_path = os.path.expanduser("~/.llama/checkpoints/Llama3.3-70B-Instruct")
    if not os.path.exists(meta_llama_path):
        print(f"❌ Meta Llama model not found at {meta_llama_path}")
        return None
    
    # First, convert to HF format
    hf_path = "./llama3_hf"
    os.makedirs(hf_path, exist_ok=True)
    
    print(f"Converting Meta model to Hugging Face format in {hf_path}...")
    try:
        # Check if we have a conversion script
        conversion_script = None
        for script in ["convert_llama3_to_hf.py", "convert_meta_to_hf.py", "convert_llama_to_hf.py"]:
            if os.path.exists(script):
                conversion_script = script
                break
        
        if not conversion_script:
            print("❌ No conversion script found (convert_llama3_to_hf.py or similar)")
            return None
        
        # Run conversion script
        cmd = f"python {conversion_script} --input-dir {meta_llama_path} --output-dir {hf_path}"
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
        
        if not os.path.exists(os.path.join(hf_path, "config.json")):
            print(f"❌ Conversion to HF format failed - no config.json in {hf_path}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return None
        
        print(f"✅ Successfully converted to HF format at {hf_path}")
        
        # Now convert to GGUF format
        # Try to locate llama.cpp repository or tools
        gguf_path = "./models/Meta-Llama-3.3-70B-Instruct.Q4_K_M.gguf"
        
        # Option 1: Use llama-cpp-python's built-in converter if available
        try:
            from llama_cpp.server.convert import convert_hf_to_gguf
            print("Using llama_cpp's built-in converter...")
            convert_hf_to_gguf(hf_path, gguf_path, quantization_level="q4_k_m")
            if os.path.exists(gguf_path):
                print(f"✅ Successfully converted to GGUF format at {gguf_path}")
                return gguf_path
        except ImportError:
            print("llama_cpp.server.convert not available, trying alternative methods...")
        except Exception as e:
            print(f"Error with llama_cpp converter: {e}")
        
        # Option 2: Try running the llama.cpp convert.py script
        try:
            # First check if we have a local convert script
            for convert_script in ["./llama.cpp/convert.py", "./convert_to_gguf.py"]:
                if os.path.exists(convert_script):
                    cmd = f"python {convert_script} --outtype q4_k_m --outfile {gguf_path} {hf_path}"
                    print(f"Running: {cmd}")
                    result = subprocess.run(cmd, shell=True, check=True)
                    if os.path.exists(gguf_path):
                        print(f"✅ Successfully converted to GGUF format at {gguf_path}")
                        return gguf_path
                    break
            
            # If no local script, try installing and using the llama-cpp-python package
            cmd = f"pip install 'llama-cpp-python[server]' && python -m llama_cpp.server.convert --outtype q4_k_m --outfile {gguf_path} {hf_path}"
            print(f"Running: {cmd}")
            result = subprocess.run(cmd, shell=True, check=True)
            if os.path.exists(gguf_path):
                print(f"✅ Successfully converted to GGUF format at {gguf_path}")
                return gguf_path
            
        except Exception as e:
            print(f"Error converting to GGUF: {e}")
            
        # If we get here, all conversion methods failed
        print("❌ Failed to convert to GGUF format")
        return None
    
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return None

def download_llama_8b_gguf():
    """Download the Llama 3 8B model in GGUF format - this is public and works."""
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Use huggingface_hub to download
    repo_id = "TheBloke/Llama-3-8B-Instruct-GGUF"
    filename = "llama-3-8b-instruct.Q4_K_M.gguf"
    output_path = os.path.join(model_dir, "Llama-3-8B-Working.Q4_K_M.gguf")
    
    print(f"Downloading {filename} from {repo_id}...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        
        if downloaded_path != output_path:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(downloaded_path, output_path)
            
        print(f"✅ Successfully downloaded model to: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error downloading from Hugging Face: {e}")
        
        # Alternative direct download URL (fallback)
        direct_url = "https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf"
        try:
            download_with_progress(direct_url, output_path)
            print(f"✅ Successfully downloaded model to: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Error with direct download: {e}")
            return None

def download_tiny_llama():
    """Download the TinyLlama model as a backup option."""
    model_dir = "./llama_gguf"
    os.makedirs(model_dir, exist_ok=True)
    
    # Use direct download for TinyLlama
    url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    output_path = os.path.join(model_dir, "tinyllama-1.1b-chat-working.Q4_K_M.gguf")
    
    try:
        download_with_progress(url, output_path)
        print(f"✅ Successfully downloaded TinyLlama to: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error downloading TinyLlama: {e}")
        return None

def update_funsearch_script(model_path):
    """Update the llama_funsearch.py script to use the working model."""
    if not model_path or not os.path.exists(model_path):
        return False
    
    # Backup the original script if not already done
    script_path = "./llama_funsearch.py"
    backup_path = "./llama_funsearch.py.original_backup"
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(script_path, backup_path)
        print(f"✅ Created backup of original script at: {backup_path}")
    
    # Read the file
    with open(script_path, "r") as f:
        content = f.read()
    
    # Find the model loading function and update the path
    updated_content = content.replace(
        "# Explicitly prioritize the 70B model",
        "# Use the working model we just downloaded"
    ).replace(
        "primary_model = './models/Llama-3.3-70B-Instruct-Q4_K_M.gguf'",
        f"primary_model = '{model_path}'"
    )
    
    # Save the updated file
    with open(script_path, "w") as f:
        f.write(updated_content)
    
    print(f"✅ Updated {script_path} to use the working model: {model_path}")
    return True

def main():
    print("=" * 70)
    print("🔽 Setting up Llama model for L-shape Ramsey Problem")
    print("=" * 70)
    
    # First try to convert Meta's Llama 3.3 70B model
    gguf_path = convert_meta_llama_to_gguf()
    
    # If that fails, download Llama 3 8B
    if not gguf_path:
        print("\nFalling back to downloading Llama 3 8B model...")
        gguf_path = download_llama_8b_gguf()
    
    # If that also fails, try TinyLlama
    if not gguf_path:
        print("\nFalling back to TinyLlama model...")
        gguf_path = download_tiny_llama()
    
    if gguf_path:
        # Update the script to use the working model
        update_funsearch_script(gguf_path)
        
        print("\n" + "=" * 70)
        print(f"✅ Successfully prepared model: {gguf_path}")
        print("You can now run: python llama_funsearch.py --grid-size 3 --iterations 2")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("❌ Failed to prepare any working model.")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())