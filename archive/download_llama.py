import os
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_llama():
    """Download Llama 3.2 8B model and tokenizer."""
    model_name = "meta-llama/Llama-2-8b-chat-hf"
    
    print("Downloading Llama 3.2 8B model...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download model and tokenizer
    model_path = snapshot_download(
        repo_id=model_name,
        local_dir="models/llama-2-8b-chat",
        local_dir_use_symlinks=False
    )
    
    print(f"Model downloaded to: {model_path}")
    
    # Load model and tokenizer to verify
    print("Verifying model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    
    print("Model and tokenizer verified successfully!")
    print("\nModel details:")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model parameters: {model.num_parameters():,}")

if __name__ == "__main__":
    download_llama() 