#!/usr/bin/env python3
"""
Update the llama_funsearch.py file to use the converted Hugging Face model.
"""

import os
import re
import argparse
import shutil

def backup_file(file_path):
    """Create a backup of the file."""
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    return backup_path

def update_model_path(file_path, new_model_path):
    """Update the model path in the file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Look for the model_path initialization in __init__
    pattern = r'(self\.model_path\s*=\s*model_path\s*or\s*)"[^"]*"'
    new_content = re.sub(pattern, f'\\1"{new_model_path}"', content)
    
    # If the pattern wasn't found or replaced, try a different approach
    if new_content == content:
        print("Could not find model_path pattern in __init__. Trying alternative pattern...")
        pattern = r'(self\.model_path\s*=\s*)(".*?"|\'.+?\')'
        new_content = re.sub(pattern, f'\\1"{new_model_path}"', content)
    
    # Add code to handle the model loading from HuggingFace format
    if new_content == content:
        print("Could not update model_path automatically. Manual edits may be required.")
    else:
        # Fix potential tokenizer loading issues
        new_content = new_content.replace(
            'self.tokenizer = AutoTokenizer.from_pretrained(',
            'try:\n            # Try newer tokenizer loading method\n            self.tokenizer = AutoTokenizer.from_pretrained('
        )
        
        new_content = new_content.replace(
            'local_files_only=True,\n                    trust_remote_code=True\n                )',
            'local_files_only=True,\n                    trust_remote_code=True\n                )\n            except Exception as e:\n                print(f"Tokenizer loading error: {e}")\n                print("Trying basic tokenizer loading...")\n                self.tokenizer = AutoTokenizer.from_pretrained(\n                    self.model_path,\n                    local_files_only=True\n                )'
        )
        
        # Write the updated content back to the file
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        print(f"Updated model path in {file_path} to {new_model_path}")
        return True
    
    return False

def main():
    """Main function to update the file."""
    parser = argparse.ArgumentParser(description="Update llama_funsearch.py to use the converted Hugging Face model")
    parser.add_argument("--file", type=str, default="llama_funsearch.py",
                        help="Path to the llama_funsearch.py file")
    parser.add_argument("--model_path", type=str, default="./llama3_hf",
                        help="Path to the converted Hugging Face model")
    parser.add_argument("--no_backup", action="store_true",
                        help="Skip creating a backup of the original file")
    
    args = parser.parse_args()
    
    # Check if the file exists
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found")
        return False
    
    # Create a backup of the file
    if not args.no_backup:
        backup_file(args.file)
    
    # Update the model path
    success = update_model_path(args.file, args.model_path)
    
    if success:
        print("\n===============================================")
        print("✅ File updated successfully!")
        print("===============================================")
        print(f"The llama_funsearch.py file now uses the model at: {args.model_path}")
        print("\nTo run with the updated model path:")
        print(f"python {args.file}")
        print("===============================================")
    else:
        print("\n===============================================")
        print("❌ File update failed.")
        print("Manual edits may be required.")
        print("===============================================")
    
    return success

if __name__ == "__main__":
    main() 