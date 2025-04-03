import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Tuple, Optional
import os
from pathlib import Path

class LShapeLlamaSolver:
    def __init__(self, model_path: str = "meta-llama/Llama-2-8b-chat-hf"):
        """Initialize the L-shape Ramsey problem solver using Llama 3.2 8B."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # System prompt for the L-shape Ramsey problem
        self.system_prompt = """You are an expert in Ramsey Theory and combinatorial optimization.
        Your task is to help solve the L-shape Ramsey problem:
        Given an n×n grid, color each cell either red or blue such that no four cells form a monochromatic L-shape.
        An L-shape is formed by three cells in a row and one cell adjacent to the middle cell of that row.
        
        Provide your solution as a Python function that returns a valid coloring.
        The function should be efficient and use clever patterns or algorithms.
        """
        
    def generate_solution(self, grid_size: int) -> Optional[np.ndarray]:
        """Generate a solution for the given grid size using Llama."""
        prompt = f"{self.system_prompt}\n\nGenerate a Python function that solves the {grid_size}×{grid_size} L-shape Ramsey problem."
        
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
            start_idx = generated_text.find("def ")
            if start_idx == -1:
                return None
                
            end_idx = generated_text.find("\n\n", start_idx)
            if end_idx == -1:
                end_idx = len(generated_text)
                
            function_text = generated_text[start_idx:end_idx]
            
            # Create a temporary file to store the function
            with open("temp_solution.py", "w") as f:
                f.write(function_text)
            
            # Import and execute the function
            import temp_solution
            solution = temp_solution.generate_grid(grid_size)
            
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
                    return False
                    
                # Check vertical L-shapes
                if grid[i,j] == grid[i+1,j] == grid[i+2,j] == grid[i+1,j+1]:
                    return False
                    
        return True
        
    def solve(self, grid_size: int, max_attempts: int = 5) -> Optional[np.ndarray]:
        """Solve the L-shape Ramsey problem for the given grid size."""
        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1}/{max_attempts}")
            solution = self.generate_solution(grid_size)
            
            if solution is not None and self.verify_solution(solution):
                return solution
                
        return None

def main():
    # Initialize solver
    solver = LShapeLlamaSolver()
    
    # Test on different grid sizes
    grid_sizes = [3, 4, 5]
    
    for size in grid_sizes:
        print(f"\nSolving {size}×{size} grid...")
        solution = solver.solve(size)
        
        if solution is not None:
            print(f"Found valid solution for {size}×{size} grid:")
            print(solution)
        else:
            print(f"No valid solution found for {size}×{size} grid")

if __name__ == "__main__":
    main() 