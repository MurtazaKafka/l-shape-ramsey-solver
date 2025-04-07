#!/usr/bin/env python3
"""
Search for Llama model files in common locations.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_find_command(base_dir, pattern):
    """Run a find command to locate model files."""
    try:
        cmd = f"find {base_dir} -name '{pattern}' -type f -not -path '*/\\.*' 2>/dev/null"
        print(f"Executing: {cmd}")
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout.strip().split('\n')
    except Exception as e:
        print(f"Error running find command: {e}")
        return []

def search_common_locations():
    """Search for Llama model files in common locations."""
    print("Searching for Llama model files...")
    
    # Common locations to search
    search_locations = [
        # User's home directory
        str(Path.home()),
        
        # Common shared locations
        "/home/DAVIDSON",  # Based on your server naming
        "/data",
        "/shared",
        "/datasets",
        "/models",
        "/scratch",
        "/work",
        
        # GPU server specific locations
        "/mnt",
        "/gpfs",
        "/lustre",
    ]
    
    # Patterns to search for
    patterns = [
        "*.bin",           # Model weights
        "config.json",     # Model configuration
        "tokenizer.json",  # Tokenizer files
        "tokenizer_config.json"
    ]
    
    # Models to look for
    model_names = [
        "llama",
        "Llama",
        "LLAMA",
        "llama3",
        "Llama3",
        "meta-llama",
        "Meta-Llama",
        "70B"
    ]
    
    all_results = []
    
    # First, try a general search for llama or 70B in model directories
    for location in search_locations:
        if not os.path.exists(location):
            continue
            
        print(f"\nSearching in {location}...")
        
        # Search for model directories
        for model_name in model_names:
            dir_cmd = f"find {location} -type d -name '*{model_name}*' 2>/dev/null"
            try:
                dir_result = subprocess.run(dir_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                directories = dir_result.stdout.strip().split('\n')
                directories = [d for d in directories if d]  # Remove empty entries
                
                if directories:
                    print(f"Found potential model directories containing '{model_name}':")
                    for directory in directories:
                        print(f"  {directory}")
                        all_results.append(directory)
            except Exception as e:
                print(f"Error searching for directories: {e}")
    
    # Then, search for specific model files
    print("\nSearching for specific model files...")
    for pattern in patterns:
        results = []
        for location in search_locations:
            if not os.path.exists(location):
                continue
                
            found_files = run_find_command(location, pattern)
            for file in found_files:
                if file and any(name.lower() in file.lower() for name in model_names):
                    results.append(file)
        
        if results:
            print(f"\nFound {len(results)} files matching '{pattern}':")
            for result in results[:5]:  # Show only first 5 to avoid overwhelming output
                print(f"  {result}")
            if len(results) > 5:
                print(f"  ... and {len(results) - 5} more")
            
            # Extract directories
            directories = set(os.path.dirname(r) for r in results)
            for directory in directories:
                if any(model_name.lower() in directory.lower() for model_name in model_names):
                    all_results.append(directory)
    
    # De-duplicate and sort results
    unique_results = sorted(set(all_results))
    
    print("\nPossible model locations (most likely candidates at the top):")
    likely_candidates = []
    for result in unique_results:
        if any(x in result.lower() for x in ["llama3", "llama-3", "llama_3"]):
            if "70b" in result.lower():
                print(f"  [MOST LIKELY] {result}")
                likely_candidates.insert(0, result)
            else:
                print(f"  [LIKELY] {result}")
                likely_candidates.append(result)
        elif "70b" in result.lower():
            print(f"  [LIKELY] {result}")
            likely_candidates.append(result)
        else:
            print(f"  {result}")
    
    print("\nRecommended actions:")
    if likely_candidates:
        print(f"1. Try running test_model_loading.py with the most likely path:")
        print(f"   python test_model_loading.py --model-path \"{likely_candidates[0]}\"")
    else:
        print("1. Check with your system administrator for the correct model path")
    
    print("2. Check if you need to load the correct environment module:")
    print("   module load llama/3.3")
    
    return unique_results

def main():
    parser = argparse.ArgumentParser(description='Search for Llama model files')
    args = parser.parse_args()
    
    search_common_locations()

if __name__ == "__main__":
    main() 