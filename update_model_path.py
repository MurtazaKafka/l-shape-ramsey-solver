#!/usr/bin/env python3
"""
Update the model path in all relevant files after finding the correct path.
"""

import os
import sys
import argparse
import re
from pathlib import Path

def update_file(file_path, old_path, new_path):
    """Update the model path in a file."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return False
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace exact path
        updated_content = content.replace(old_path, new_path)
        
        # Also try to replace the path in default parameters
        pattern = r'(model_path\s*=\s*(?:None|").*?)' + re.escape(old_path.split('/')[-1]) + r'(".*?)'
        updated_content = re.sub(pattern, f'\\1{new_path.split("/")[-1]}\\2', updated_content)
        
        # If no changes, try a more general approach
        if content == updated_content:
            pattern = r'(model_path\s*=\s*")([^"]*?)(")'
            updated_content = re.sub(pattern, f'\\1{new_path}\\3', updated_content)
        
        if content != updated_content:
            with open(file_path, 'w') as f:
                f.write(updated_content)
            print(f"Updated {file_path} with new model path.")
            return True
        else:
            print(f"No changes made to {file_path} (path not found).")
            return False
    
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def update_all_files(new_path):
    """Update the model path in all relevant files."""
    # Default old path to replace
    old_path = "/home/DAVIDSON/murtaza/.llama/checkpoints/Llama3.3-70B-Instruct"
    
    files_to_update = [
        "llama_funsearch.py",
        "test_model_loading.py"
    ]
    
    success_count = 0
    
    for file in files_to_update:
        if update_file(file, old_path, new_path):
            success_count += 1
    
    print(f"\nUpdated {success_count} of {len(files_to_update)} files with new model path: {new_path}")
    
    # Additional instructions
    print("\nNext steps:")
    print("1. Run test_model_loading.py to verify the model can be loaded:")
    print(f"   python test_model_loading.py")
    print("2. Then run llama_funsearch.py to solve the L-shape Ramsey problem:")
    print("   python llama_funsearch.py --grid-size 3")

def main():
    parser = argparse.ArgumentParser(description='Update model path in all relevant files')
    parser.add_argument('model_path', help='New path to the Llama model')
    args = parser.parse_args()
    
    # Validate the input path
    if not os.path.exists(args.model_path):
        print(f"Warning: The specified path {args.model_path} does not exist.")
        confirm = input("Do you still want to proceed with updating the files? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)
    
    update_all_files(args.model_path)

if __name__ == "__main__":
    main() 