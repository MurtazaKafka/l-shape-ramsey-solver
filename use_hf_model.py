import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Test grid coloring with L-shape Ramsey problem")
    parser.add_argument("--grid_size", type=int, default=4, help="Size of the grid")
    parser.add_argument("--colors", type=int, default=3, help="Number of colors")
    parser.add_argument("--iterations", type=int, default=5, help="Number of FunSearch iterations")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for model generation")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Using HuggingFace Llama model instead of local model")
    print("Loading model from HuggingFace Hub...")
    
    # Use a compatible Llama model from HuggingFace Hub
    model_name = "meta-llama/Llama-2-7b-hf"  # Smaller model for testing
    
    print(f"Using model: {model_name}")
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB total")
        device = "cuda"
    else:
        print("No GPU detected, using CPU (this will be very slow)...")
        device = "cpu"
    
    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        
        # Test generation
        prompt = "Solve the L-shape Ramsey problem for a 4x4 grid with 3 colors."
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        print("\nTesting generation with prompt:", prompt)
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=100,
            temperature=args.temperature,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\nModel response:", response)
        
        print("\nModel loaded successfully!")
        print("You can now implement the FunSearch algorithm using this model.")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return

if __name__ == "__main__":
    main()
