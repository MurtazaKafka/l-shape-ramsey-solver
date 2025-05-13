#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import importlib.util

ALTERNATIVE_MODELS = {
    "phi3": {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "description": "Microsoft Phi-3 Mini (3.8B parameters)"
    },
    "gemma2": {
        "name": "google/gemma-2-2b-it",
        "description": "Google Gemma 2 (2B parameters)"
    },
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "description": "Mistral AI 7B Instruct"
    }
}

class FreeModelSolver:
    def __init__(self, model_choice="phi3"):
        """Initialize with an open model that doesn't require special permissions."""
        if model_choice not in ALTERNATIVE_MODELS:
            print(f"Model {model_choice} not found. Using phi3 as default.")
            model_choice = "phi3"
            
        model_info = ALTERNATIVE_MODELS[model_choice]
        model_name = model_info["name"]
        
        print(f"Using {model_info['description']} model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.model_choice = model_choice
        
        # System prompt for the L-shape Ramsey problem
        self.system_prompt = """You are an expert in Ramsey Theory and combinatorial optimization.
Your task is to help solve the L-shape Ramsey problem:
Given an n×n grid, color each cell either red or blue such that no four cells form a monochromatic L-shape.
An L-shape is formed by three cells in a row and one cell adjacent to the middle cell of that row.

Provide your solution as a Python function named 'generate_grid' that takes grid_size as input and returns a numpy array with values 0 (red) and 1 (blue).
The function should be efficient and use clever patterns or algorithms."""
        
    def generate_solution(self, grid_size: int) -> Optional[np.ndarray]:
        """Generate a solution for the given grid size using the model."""
        
        user_message = f"Write a Python function that solves the {grid_size}×{grid_size} L-shape Ramsey problem. The function should be named 'generate_grid' and return a numpy array with 0's and 1's representing the colors."
        
        # Format prompt based on model
        if self.model_choice == "phi3":
            prompt = f"<|system|>\n{self.system_prompt}\n<|user|>\n{user_message}\n<|assistant|>"
        elif self.model_choice == "gemma2":
            prompt = f"<start_of_turn>system\n{self.system_prompt}<end_of_turn>\n<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"
        else:  # Default for Mistral and others
            prompt = f"<s>[INST] {self.system_prompt} [/INST]</s>\n[INST] {user_message} [/INST]"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the Python function from the generated text
        try:
            # Find the function definition
            start_idx = generated_text.find("def generate_grid")
            if start_idx == -1:
                print("No function found in generated text.")
                print("Generated text snippet:")
                print(generated_text[:500] + "..." if len(generated_text) > 500 else generated_text)
                return None
                
            # Find the end of the function
            # Look for markers that might indicate the end of the function
            end_markers = ["```", "def ", "# Test", "# Example", "if __name__"]
            end_positions = [generated_text.find(marker, start_idx + 15) for marker in end_markers]
            end_positions = [pos for pos in end_positions if pos != -1]
            
            if end_positions:
                end_idx = min(end_positions)
            else:
                end_idx = len(generated_text)
                
            function_text = generated_text[start_idx:end_idx].strip()
            function_text = function_text.replace("```python", "").replace("```", "")
            
            print(f"Extracted function:\n{function_text}\n")
            
            # Create a temporary file to store the function
            with open("temp_solution.py", "w") as f:
                f.write(function_text)
            
            # Import and execute the function
            spec = importlib.util.spec_from_file_location("temp_solution", "temp_solution.py")
            temp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_module)
            
            solution = temp_module.generate_grid(grid_size)
            
            # Clean up
            os.remove("temp_solution.py")
            
            return solution
            
        except Exception as e:
            print(f"Error executing generated solution: {e}")
            return None
            
    def verify_solution(self, grid: np.ndarray) -> bool:
        """Verify if a given grid coloring is valid (no monochromatic L-shapes)."""
        n = len(grid)
        
        # Check all possible L-shapes
        for i in range(n-2):
            for j in range(n-1):
                # Check horizontal L-shapes
                if grid[i,j] == grid[i,j+1] == grid[i,j+2] == grid[i+1,j+1]:
                    print(f"Found invalid horizontal L-shape at ({i},{j})")
                    return False
                    
                # Check vertical L-shapes
                if grid[i,j] == grid[i+1,j] == grid[i+2,j] == grid[i+1,j+1]:
                    print(f"Found invalid vertical L-shape at ({i},{j})")
                    return False
                    
                if j > 0:
                    # Check rotated L-shapes
                    if grid[i,j] == grid[i+1,j] == grid[i+2,j] == grid[i+1,j-1]:
                        print(f"Found invalid rotated L-shape at ({i},{j})")
                        return False
                        
        # Check rotated horizontal L-shapes
        for i in range(1, n):
            for j in range(n-2):
                if grid[i,j] == grid[i,j+1] == grid[i,j+2] == grid[i-1,j+1]:
                    print(f"Found invalid rotated horizontal L-shape at ({i},{j})")
                    return False
                    
        return True
        
    def solve(self, grid_size: int, max_attempts: int = 10) -> Optional[np.ndarray]:
        """Solve the L-shape Ramsey problem for the given grid size."""
        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1}/{max_attempts}")
            solution = self.generate_solution(grid_size)
            
            if solution is not None:
                print(f"Generated solution shape: {solution.shape}")
                if self.verify_solution(solution):
                    self.visualize_solution(solution, grid_size)
                    return solution
                else:
                    print("Solution failed verification.")
                    
        return None
        
    def visualize_solution(self, grid: np.ndarray, grid_size: int):
        """Visualize the solution grid."""
        plt.figure(figsize=(8, 8))
        cmap = plt.cm.colors.ListedColormap(['red', 'blue'])
        plt.imshow(grid, cmap=cmap)
        plt.grid(True, color='black', linewidth=1.5)
        plt.xticks(range(grid_size))
        plt.yticks(range(grid_size))
        plt.title(f"L-shape Ramsey {grid_size}×{grid_size} Grid Solution")
        
        # Save the visualization
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig(f"visualizations/l_shape_grid_{grid_size}x{grid_size}_{self.model_choice}.png")
        plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Solve L-shape Ramsey problem using free models')
    parser.add_argument('--model', choices=list(ALTERNATIVE_MODELS.keys()), default='phi3',
                      help='Model to use (default: phi3)')
    parser.add_argument('--grid-sizes', type=int, nargs='+', default=[3, 4, 5, 6],
                      help='Grid sizes to solve (default: 3 4 5 6)')
    args = parser.parse_args()
    
    # Initialize solver
    solver = FreeModelSolver(model_choice=args.model)
    
    # Test on different grid sizes
    grid_sizes = args.grid_sizes
    
    results = {}
    
    for size in grid_sizes:
        print(f"\nSolving {size}×{size} grid...")
        solution = solver.solve(size)
        
        if solution is not None:
            print(f"Found valid solution for {size}×{size} grid:")
            print(solution)
            results[size] = "Solved"
        else:
            print(f"No valid solution found for {size}×{size} grid")
            results[size] = "Unsolved"
    
    print("\nSummary of results:")
    for size, result in results.items():
        print(f"{size}×{size} grid: {result}")

if __name__ == "__main__":
    main() 