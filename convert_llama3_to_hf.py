#!/usr/bin/env python3
"""
Convert Meta's original Llama 3 model format to Hugging Face format.
This modified script specifically handles Llama 3 models.
"""

import os
import sys
import json
import torch
import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaTokenizerFast, AutoTokenizer

# Define sharding configuration for different model sizes
NUM_SHARDS = {
    # Original Llama models
    "7B": 1,
    "13B": 2,
    "30B": 4,
    "65B": 8,
    "70B": 8,
    # Llama 2 models
    "Llama-2-7b": 1,
    "Llama-2-7b-chat": 1,
    "Llama-2-13b": 2,
    "Llama-2-13b-chat": 2,
    "Llama-2-70b": 8,
    "Llama-2-70b-chat": 8,
    # Llama 3 models
    "Llama-3-8B": 1,
    "Llama-3-8B-Instruct": 1,
    "Llama-3-70B": 8,
    "Llama-3-70B-Instruct": 8,
    # Common variations
    "7b": 1,
    "13b": 2,
    "30b": 4,
    "65b": 8,
    "70b": 8,
}

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def write_model(
    model_path: str,
    input_base_path: str,
    model_size: str,
    tokenizer_path: Optional[str] = None,
    num_shards: Optional[int] = None,
    use_safetensors: bool = True,
    push_to_hub: bool = False,
):
    # Find the number of shards for this model size
    if model_size not in NUM_SHARDS and model_size.lower() not in NUM_SHARDS:
        print(f"Model size {model_size} not found in NUM_SHARDS dictionary")
        print(f"Available sizes: {list(NUM_SHARDS.keys())}")
        print(f"Fallback to default size: 70B")
        model_size_for_shards = "70B"
    else:
        model_size_for_shards = model_size if model_size in NUM_SHARDS else model_size.lower()
    
    # Set number of shards based on model size or provided value
    num_shards = NUM_SHARDS[model_size_for_shards] if num_shards is None else num_shards
    
    # Load and convert params.json to config
    params_path = os.path.join(input_base_path, "params.json")
    if os.path.exists(params_path):
        params = read_json(params_path)
        print(f"Loaded params from {params_path}: {params}")
        
        n_layers = params.get("n_layers", 32)  # Default for 7B model
        n_heads = params.get("n_heads", 32)
        dim = params.get("dim", 4096)
        intermediate_size = params.get("hidden_dim", dim * 4)
        
        # Determine vocab size
        vocab_size = params.get("vocab_size", 32000)  # Default Llama 2 vocab size
    else:
        print(f"No params.json found at {params_path}, using defaults for Llama 3 70B")
        # Defaults for Llama 3 70B
        n_layers = 80
        n_heads = 64
        dim = 8192
        intermediate_size = dim * 4
        vocab_size = 128256  # Llama 3 vocab size
    
    # Special handling for Llama 3 models
    if "llama-3" in model_size.lower() or "llama3" in model_size.lower():
        # Adjust parameters for Llama 3
        if "70b" in model_size.lower() or "70B" in model_size.lower():
            n_layers = 80
            n_heads = 64
            dim = 8192
            vocab_size = 128256  # Llama 3 vocab size
        elif "8b" in model_size.lower() or "8B" in model_size.lower():
            n_layers = 32
            n_heads = 32
            dim = 4096
            vocab_size = 128256  # Llama 3 vocab size
    
    # Create the config
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=dim,
        intermediate_size=intermediate_size,
        num_attention_heads=n_heads,
        num_hidden_layers=n_layers,
        rms_norm_eps=1e-5,
        max_position_embeddings=4096,  # Llama 3 supports 8K context, but 4K is more compatible
        architectures=["LlamaForCausalLM"],
        initializer_range=0.02,
        model_type="llama",
    )
    
    # Create the model directory
    os.makedirs(model_path, exist_ok=True)
    
    # Save the config
    config.save_pretrained(model_path)
    print(f"Saved config to {model_path}")
    
    # Get list of consolidated model files
    consolidated_files = sorted([f for f in os.listdir(input_base_path) if f.startswith("consolidated.")])
    if not consolidated_files:
        print(f"No consolidated files found in {input_base_path}")
        print(f"Files in directory: {os.listdir(input_base_path)}")
        return
    
    print(f"Found {len(consolidated_files)} consolidated files: {consolidated_files}")
    
    # Copy tokenizer
    if tokenizer_path is None:
        tokenizer_path = os.path.join(input_base_path, "tokenizer.model")
    
    if os.path.exists(tokenizer_path):
        # Try using AutoTokenizer first
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            tokenizer.save_pretrained(model_path)
            print(f"Saved tokenizer to {model_path} using AutoTokenizer")
        except Exception as e:
            print(f"Error loading tokenizer with AutoTokenizer: {e}")
            print("Trying LlamaTokenizerFast...")
            try:
                tokenizer = LlamaTokenizerFast(tokenizer_file=tokenizer_path)
                tokenizer.save_pretrained(model_path)
                print(f"Saved tokenizer to {model_path} using LlamaTokenizerFast")
            except Exception as e2:
                print(f"Error loading tokenizer with LlamaTokenizerFast: {e2}")
                print("Copying tokenizer files manually...")
                if os.path.exists(tokenizer_path):
                    import shutil
                    dest_path = os.path.join(model_path, "tokenizer.model")
                    shutil.copy(tokenizer_path, dest_path)
                    print(f"Copied tokenizer file from {tokenizer_path} to {dest_path}")
    else:
        print(f"Tokenizer not found at {tokenizer_path}")
    
    # Load and convert weights
    print(f"Converting weights to {num_shards} shard(s)...")
    
    # Load consolidated files
    state_dict = {}
    for file in tqdm(consolidated_files, desc="Loading model files"):
        file_path = os.path.join(input_base_path, file)
        weights = torch.load(file_path, map_location="cpu")
        state_dict.update(weights)
    
    # Convert weights to HF format
    hf_state_dict = {}
    
    # For embedding layer
    if "tok_embeddings.weight" in state_dict:
        hf_state_dict["model.embed_tokens.weight"] = state_dict["tok_embeddings.weight"]
    
    # For normalization layers
    if "norm.weight" in state_dict:
        hf_state_dict["model.norm.weight"] = state_dict["norm.weight"]
    
    # For output layer
    if "output.weight" in state_dict:
        hf_state_dict["lm_head.weight"] = state_dict["output.weight"]
    
    # For model layers
    layers_pattern = re.compile(r"layers\.(\d+)\.([^.]+)\.([^.]+)")
    for name, param in tqdm(state_dict.items(), desc="Converting weights"):
        # Skip already processed weights
        if name in ["tok_embeddings.weight", "norm.weight", "output.weight"]:
            continue
        
        match = layers_pattern.match(name)
        if match:
            layer_idx, module_type, param_name = match.groups()
            
            # Convert layer weights
            if module_type == "attention":
                # Handle attention module
                if param_name == "wq.weight":
                    hf_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = param
                elif param_name == "wk.weight":
                    hf_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = param
                elif param_name == "wv.weight":
                    hf_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = param
                elif param_name == "wo.weight":
                    hf_state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = param
            elif module_type == "feed_forward":
                # Handle feed forward module
                if param_name == "w1.weight":
                    hf_state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = param
                elif param_name == "w2.weight":
                    hf_state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = param
                elif param_name == "w3.weight":
                    hf_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = param
            elif module_type == "attention_norm":
                # Handle attention normalization
                if param_name == "weight":
                    hf_state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = param
            elif module_type == "ffn_norm":
                # Handle feed forward normalization
                if param_name == "weight":
                    hf_state_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = param
        else:
            print(f"Unmatched parameter: {name}")
    
    # Split the state dict for sharding
    if num_shards > 1:
        state_dict_shards = [{} for _ in range(num_shards)]
        
        # Process each parameter
        for name, param in tqdm(hf_state_dict.items(), desc="Sharding weights"):
            if name.startswith("model.layers."):
                # For layer weights, shard by layer
                layer_idx = int(name.split(".")[2])
                shard_idx = layer_idx % num_shards
                state_dict_shards[shard_idx][name] = param
            else:
                # For global weights, put in first shard
                state_dict_shards[0][name] = param
    else:
        state_dict_shards = [hf_state_dict]
    
    # Save weights in shards
    for i, shard in enumerate(state_dict_shards):
        if use_safetensors:
            shard_path = os.path.join(model_path, f"model-{i:05d}-of-{num_shards:05d}.safetensors")
            save_file(shard, shard_path)
        else:
            shard_path = os.path.join(model_path, f"pytorch_model-{i:05d}-of-{num_shards:05d}.bin")
            torch.save(shard, shard_path)
        print(f"Saved shard {i+1}/{num_shards} to {shard_path}")
    
    # Save model index file
    index = {"metadata": {"total_size": sum(param.numel() * param.element_size() for param in hf_state_dict.values())}}
    if use_safetensors:
        index["weight_map"] = {name: f"model-{i:05d}-of-{num_shards:05d}.safetensors" 
                               for i, shard in enumerate(state_dict_shards) 
                               for name in shard.keys()}
    else:
        index["weight_map"] = {name: f"pytorch_model-{i:05d}-of-{num_shards:05d}.bin" 
                               for i, shard in enumerate(state_dict_shards) 
                               for name in shard.keys()}
    
    index_path = os.path.join(model_path, "pytorch_model.bin.index.json")
    write_json(index, index_path)
    print(f"Saved model index to {index_path}")
    
    print(f"Model conversion completed successfully!")
    
    # Push to hub if requested
    if push_to_hub:
        from huggingface_hub import create_repo, upload_folder
        
        repo_name = os.path.basename(model_path)
        create_repo(repo_name, exist_ok=True)
        upload_folder(folder_path=model_path, repo_id=repo_name)
        print(f"Pushed model to hub: {repo_name}")

def main():
    parser = argparse.ArgumentParser(description="Convert Llama 3 weights to Hugging Face format")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing Meta's model files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory for the converted model")
    parser.add_argument("--model_size", type=str, default="Llama-3-70B", help="Model size, e.g., 'Llama-3-70B', 'Llama-3-8B'")
    parser.add_argument("--num_shards", type=int, default=None, help="Number of shards to split the model into")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the tokenizer file")
    parser.add_argument("--use_safetensors", action="store_true", help="Use safetensors format instead of PyTorch")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the converted model to Hugging Face Hub")
    
    args = parser.parse_args()
    
    # Print conversion parameters
    print(f"Converting model from {args.input_dir} to {args.output_dir}")
    print(f"Model size: {args.model_size}")
    print(f"Number of shards: {args.num_shards or 'auto'}")
    print(f"Using safetensors: {args.use_safetensors}")
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Perform the conversion
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        model_size=args.model_size,
        tokenizer_path=args.tokenizer_path,
        num_shards=args.num_shards,
        use_safetensors=args.use_safetensors,
        push_to_hub=args.push_to_hub,
    )

if __name__ == "__main__":
    main() 