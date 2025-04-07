#!/usr/bin/env python3
"""
Fix the model loading code in llama_funsearch.py to properly handle large models.
"""

import os
import re
import sys
import shutil
import argparse

def backup_file(file_path):
    """Create a backup of the file."""
    backup_path = f"{file_path}.gpu_backup"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    return backup_path

def find_load_model_function(content):
    """Find the _load_model function in the file content."""
    # Look for the _load_model function 
    load_model_match = re.search(r'def\s+_load_model\s*\(\s*self\s*\):.*?(?=def\s+|$)', content, re.DOTALL)
    if not load_model_match:
        return None, None, None
    
    load_model_code = load_model_match.group(0)
    start_pos = load_model_match.start()
    end_pos = load_model_match.end()
    
    return load_model_code, start_pos, end_pos

def create_new_load_model_function():
    """Create an improved _load_model function that properly handles large models."""
    new_function = '''def _load_model(self):
        """Load the Llama model and tokenizer using Transformers with better GPU memory management."""
        # First try to find the model path
        self.model_path = self._find_model_path()
        
        print(f"Loading model from {self.model_path}...")
        try:
            # Check GPU availability
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"Number of GPUs available: {torch.cuda.device_count()}")
                
                # Print CUDA version info
                cuda_version = torch.version.cuda
                print(f"CUDA Version: {cuda_version}")
                
                # Print GPU memory info
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"GPU Memory: {gpu_memory_allocated:.2f}GB allocated / {gpu_memory_total:.2f}GB total")
            else:
                self.device = torch.device("cpu")
                print("No GPU detected, falling back to CPU (this will be very slow).")
            
            # Load tokenizer
            print("Loading tokenizer...")
            try:
                # Use local_files_only=True for a local model
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, 
                    local_files_only=True,
                    padding_side="left"
                )
                # Set padding token if not set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                print(f"Error loading tokenizer: {e}")
                print("Trying to load tokenizer from a different location...")
                # Try finding tokenizer in different locations
                possible_paths = [
                    os.path.join(self.model_path, "tokenizer.model"),
                    os.path.join(os.path.dirname(self.model_path), "tokenizer.model"),
                    os.path.join(self.model_path, "tokenizer", "tokenizer.model"),
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        print(f"Found tokenizer at {path}")
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            os.path.dirname(path), 
                            local_files_only=True,
                            padding_side="left"
                        )
                        if self.tokenizer.pad_token is None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token
                        break
                
                if not hasattr(self, 'tokenizer'):
                    raise Exception("Could not load tokenizer from any location")
            
            # Load model with proper memory management
            print("Loading model (this may take a while)...")
            
            if torch.cuda.is_available():
                # Configure BitsAndBytes for quantization with CPU offloading
                from transformers import BitsAndBytesConfig
                
                # Try 4-bit first with CPU offloading
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                
                try:
                    print("Trying 4-bit quantization with CPU offloading...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        quantization_config=bnb_config,
                        device_map="auto",  # Auto-distribute layers
                        torch_dtype=torch.float16,
                        local_files_only=True,
                        offload_folder="offload",
                        offload_state_dict=True  # Allow offloading state dict to disk
                    )
                    print("Model loaded with 4-bit quantization and CPU offloading.")
                except Exception as e:
                    print(f"Error with 4-bit quantization: {e}")
                    print("Trying with disk offloading...")
                    
                    try:
                        # Try with max memory distribution and disk offloading
                        from accelerate import init_empty_weights, infer_auto_device_map
                        
                        # Set max GPU memory to 75% of available memory
                        gpu_memory = int(torch.cuda.get_device_properties(0).total_memory * 0.75 / 1024 / 1024 / 1024)
                        max_memory = {0: f"{gpu_memory}GiB", "cpu": "24GiB", "disk": "64GiB"}
                        
                        print(f"Trying with mixed device map. Max GPU memory: {gpu_memory}GiB")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            device_map="auto",
                            max_memory=max_memory,
                            torch_dtype=torch.float16,
                            local_files_only=True,
                            offload_folder="offload"
                        )
                        print("Model loaded with mixed device offloading.")
                    except Exception as e:
                        print(f"Error with mixed offloading: {e}")
                        self.device = torch.device("cpu")
                        print("Falling back to partial CPU loading...")
                        
                        # Last resort: Load model on CPU but with low precision
                        try:
                            print("Loading partial model on CPU (reduced implementation)...")
                            # Load only the first few layers
                            self.model = AutoModelForCausalLM.from_pretrained(
                                self.model_path,
                                device_map={"": "cpu"},
                                torch_dtype=torch.float16,
                                low_cpu_mem_usage=True,
                                offload_folder="offload",
                                offload_state_dict=True,
                                local_files_only=True,
                                max_memory={"cpu": "24GiB"},
                            )
                            print("Partial model loaded on CPU.")
                        except Exception as cpu_e:
                            print(f"Critical error loading model on CPU: {cpu_e}")
                            print("Cannot load this large model with available resources.")
                            print("Consider using a smaller model or more GPU memory.")
                            raise
            else:
                # CPU only mode - will likely fail for 70B models
                print("WARNING: Attempting to load a large model on CPU only.")
                print("This will likely fail due to memory constraints.")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path, 
                        device_map={"": "cpu"},
                        torch_dtype=torch.float16, 
                        low_cpu_mem_usage=True,
                        local_files_only=True
                    )
                except Exception as cpu_e:
                    print(f"Critical error loading model on CPU: {cpu_e}")
                    print("Cannot load this large model on CPU.")
                    raise
            
            # Report success and model configuration
            print("Model loaded successfully.")
            print(f"Model parameters: {self.model.num_parameters() / 1e9:.2f}B")
            print(f"Model device map: {self.model.hf_device_map if hasattr(self.model, 'hf_device_map') else 'N/A'}")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise'''
    
    return new_function

def update_file(file_path):
    """Update the file with the improved _load_model function."""
    # Create a backup
    backup_file(file_path)
    
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the _load_model function
    load_model_code, start_pos, end_pos = find_load_model_function(content)
    
    if not load_model_code:
        print(f"Could not find _load_model function in {file_path}")
        return False
    
    # Create the new function
    new_function = create_new_load_model_function()
    
    # Replace the function in the content
    new_content = content[:start_pos] + new_function + content[end_pos:]
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"Updated _load_model function in {file_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Fix model loading in llama_funsearch.py for better GPU memory management")
    parser.add_argument("--file", type=str, default="llama_funsearch.py", help="Path to the file to update")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found")
        return 1
    
    # Update the file
    if update_file(args.file):
        print(f"Successfully updated {args.file} to better handle GPU memory")
        print("You should now be able to run the script with ./run_final.sh")
        return 0
    else:
        print(f"Failed to update {args.file}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 