#!/usr/bin/env python3
"""
Model downloader for L-Shape Ramsey Solver

This script helps download the necessary model files for the L-Shape Ramsey Solver.
It provides a simple interface to download Llama 3.3 and TinyLlama models
from Hugging Face Hub, with progress tracking and validation.
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("model_downloader")

# Model configuration
MODELS = {
    "llama3-70b": {
        "repo_id": "meta-llama/Llama-3.3-70B-Instruct-GGUF",
        "filename": "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "description": "Llama 3.3 70B (Quantized 4-bit)",
        "size_gb": "~40GB"
    },
    "tinyllama": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
        "description": "TinyLlama 1.1B (Quantized 4-bit)",
        "size_gb": "~700MB"
    }
}

def download_model(model_key, output_dir="models", force=False):
    """
    Download a model from Hugging Face Hub
    
    Args:
        model_key: Key from the MODELS dictionary
        output_dir: Directory to save the model
        force: Whether to force re-download if the file exists
    
    Returns:
        Path to the downloaded model file
    """
    if model_key not in MODELS:
        logger.error(f"Unknown model: {model_key}")
        logger.info(f"Available models: {', '.join(MODELS.keys())}")
        return None
    
    model_info = MODELS[model_key]
    model_path = os.path.join(output_dir, model_info["filename"])
    
    # Check if the model already exists
    if os.path.exists(model_path) and not force:
        logger.info(f"Model already exists at {model_path}")
        return model_path
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Downloading {model_info['description']} ({model_info['size_gb']})")
    logger.info(f"From {model_info['repo_id']}")
    
    try:
        path = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["filename"],
            local_dir=output_dir,
        )
        logger.info(f"Successfully downloaded model to {path}")
        return path
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        logger.info("Make sure you're logged in to Hugging Face Hub with `huggingface-cli login`")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download models for L-Shape Ramsey Solver")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all",
                        help="Model to download (default: all)")
    parser.add_argument("--output-dir", default="models", help="Directory to save models")
    parser.add_argument("--force", action="store_true", help="Force re-download even if model exists")
    
    args = parser.parse_args()
    
    # Print available models
    print("Available models:")
    for key, info in MODELS.items():
        print(f"  {key}: {info['description']} ({info['size_gb']})")
    print()
    
    if args.model == "all":
        for model_key in MODELS:
            download_model(model_key, args.output_dir, args.force)
    else:
        download_model(args.model, args.output_dir, args.force)
    
    print("\nTo use these models with the L-Shape Ramsey Solver, run:")
    print(f"  python llama_funsearch.py --model-path {args.output_dir}/{MODELS['llama3-70b']['filename']} --grid-size 5")
    print("  or for the smaller model:")
    print(f"  python llama_funsearch.py --model-path {args.output_dir}/{MODELS['tinyllama']['filename']} --grid-size 4")

if __name__ == "__main__":
    main()
