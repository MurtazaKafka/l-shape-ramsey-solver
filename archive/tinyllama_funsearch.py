#!/usr/bin/env python3
"""
L-shape Ramsey Problem solver using FunSearch with TinyLlama
"""

import argparse
import logging
import random
import sys
import time
import os
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Callable, Any
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

class TinyLlamaFunSearch:
    """
    Implementation of FunSearch algorithm using TinyLlama for the L-shape Ramsey problem
    """
    
    def __init__(
        self,
        model_path: str = None,
        temperature: float = 0.8,
    ):
        """
        Initialize FunSearch with TinyLlama
        
        Args:
            model_path: Path to the model
            temperature: Temperature for text generation
        """
        self.model_path = model_path or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.temperature = temperature
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """
        Load the TinyLlama model and tokenizer
        """
        try:
            # Check if PyTorch is properly installed
            print(f"PyTorch successfully imported, version: {torch.__version__}")
            
            # Check if transformers is properly installed
            import transformers
            print("Transformers successfully imported")
            
            # Check for GPU
            if torch.cuda.is_available():
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"Number of GPUs available: {torch.cuda.device_count()}")
                print(f"CUDA Version: {torch.version.cuda}")
                print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB total")
            else:
                print("No GPU detected, using CPU (this will be very slow)...")
            
            print(f"Loading model from {self.model_path}...")
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            
            print("Model and tokenizer loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def generate_python_function(self, prompt: str, max_new_tokens: int = 1024) -> str:
        """
        Generate Python function based on the prompt
        
        Args:
            prompt: The prompt for the model
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated Python function as a string
        """
        try:
            # Format the prompt for chat model
            model_input = f"<|user|>\n{prompt}\n<|assistant|>\n"
            
            # Tokenize input
            inputs = self.tokenizer(model_input, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            response = generated_text.split("<|assistant|>")[-1].strip()
            
            # Try to extract code between ```python and ``` markers
            code_blocks = []
            in_code_block = False
            code_lines = []
            
            for line in response.split("\n"):
                if line.strip().startswith("```python"):
                    in_code_block = True
                    continue
                elif line.strip() == "```" and in_code_block:
                    in_code_block = False
                    code_blocks.append("\n".join(code_lines))
                    code_lines = []
                    continue
                
                if in_code_block:
                    code_lines.append(line)
            
            # If code blocks found, use the first one
            if code_blocks:
                return code_blocks[0]
            
            # Otherwise, return the whole response as it might be code without markers
            return response
            
        except Exception as e:
            logger.error(f"Error in generate_python_function: {e}")
            return ""

    def evaluate_coloring(self, grid: List[List[int]], grid_size: int, num_colors: int) -> bool:
        """
        Evaluate whether a grid coloring is valid (no monochromatic L-shapes)
        
        Args:
            grid: 2D grid of colors
            grid_size: Size of the grid
            num_colors: Number of colors used
            
        Returns:
            True if the coloring is valid (no monochromatic L-shapes), False otherwise
        """
        # Check all possible L-shapes in the grid
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                # Check L-shape: (i,j), (i+1,j), (i,j+1)
                if grid[i][j] == grid[i+1][j] == grid[i][j+1]:
                    return False
                
                # Check L-shape: (i+1,j+1), (i,j+1), (i+1,j)
                if grid[i+1][j+1] == grid[i][j+1] == grid[i+1][j]:
                    return False
        
        return True

    def run_function(self, func_str: str, grid_size: int, num_colors: int) -> Tuple[bool, List[List[int]]]:
        """
        Run the generated function to produce a grid coloring
        
        Args:
            func_str: String containing Python function code
            grid_size: Size of the grid
            num_colors: Number of colors to use
            
        Returns:
            Tuple of (success, grid)
        """
        # Create namespace for execution
        namespace = {
            'np': np,
            'random': random,
            'grid_size': grid_size,
            'num_colors': num_colors
        }
        
        try:
            # Execute the function in the namespace
            exec(func_str, namespace)
            
            # Look for a function that generates colorings
            for func_name in ['create_grid', 'generate_coloring', 'color_grid']:
                if func_name in namespace:
                    grid = namespace[func_name](grid_size, num_colors)
                    
                    # Convert to list of lists if it's a numpy array
                    if isinstance(grid, np.ndarray):
                        grid = grid.tolist()
                    
                    # Check dimensions
                    if len(grid) != grid_size or any(len(row) != grid_size for row in grid):
                        logger.warning(f"Generated grid has incorrect dimensions: {len(grid)}x{len(grid[0]) if grid else 0}, expected {grid_size}x{grid_size}")
                        return False, []
                    
                    # Check if colors are valid (between 0 and num_colors-1)
                    if any(any(color < 0 or color >= num_colors for color in row) for row in grid):
                        logger.warning("Generated grid has invalid colors")
                        return False, []
                    
                    # Evaluate the coloring
                    valid = self.evaluate_coloring(grid, grid_size, num_colors)
                    return valid, grid
            
            # If no known function was found, look for any function that returns a 2D list or array
            for func_name, func in namespace.items():
                if callable(func) and func_name not in ['exec', 'eval']:
                    try:
                        result = func(grid_size, num_colors)
                        if isinstance(result, (list, np.ndarray)):
                            # Convert to list of lists if it's a numpy array
                            if isinstance(result, np.ndarray):
                                result = result.tolist()
                            
                            # Check dimensions
                            if len(result) == grid_size and all(len(row) == grid_size for row in result):
                                valid = self.evaluate_coloring(result, grid_size, num_colors)
                                return valid, result
                    except Exception:
                        pass
            
            logger.warning("No suitable function found in generated code")
            return False, []
            
        except Exception as e:
            logger.error(f"Error executing generated function: {e}")
            return False, []

    def generate_prompt(self, grid_size: int, num_colors: int, successful_examples: List[Dict[str, Any]] = None) -> str:
        """
        Generate a prompt for the model to create a grid coloring function
        
        Args:
            grid_size: Size of the grid
            num_colors: Number of colors to use
            successful_examples: List of successful examples to include in the prompt
            
        Returns:
            Prompt string
        """
        prompt = f"""
Write a Python function that generates a valid coloring for a {grid_size}x{grid_size} grid using {num_colors} colors (0 to {num_colors-1}).

An L-shape consists of three cells in the shape of an L:
- (i,j), (i+1,j), (i,j+1) or 
- (i+1,j+1), (i,j+1), (i+1,j)

A valid coloring means that no L-shape can have all three cells colored with the same color.

Return the grid as a list of lists, where each inner list represents a row of the grid.

Example function for a small grid:

```python
def create_grid(grid_size, num_colors):
    # Initialize grid with zeros
    grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Add your coloring logic here
    
    return grid
```

Your function should ensure no L-shape has all three cells of the same color.
"""
        
        # Add successful examples if available
        if successful_examples:
            prompt += "\n\nHere are some successful examples:\n\n"
            for i, example in enumerate(successful_examples[-2:]):  # Only show the last 2 examples
                prompt += f"Example {i+1}:\n```python\n{example['code']}\n```\n\n"
                
        prompt += "Now implement a function to create a valid coloring for a {grid_size}x{grid_size} grid with {num_colors} colors, ensuring no L-shape has all three cells of the same color:"
        
        return prompt

    def visualize_grid(self, grid: List[List[int]], grid_size: int, num_colors: int, title: str = None) -> None:
        """
        Visualize a colored grid
        
        Args:
            grid: 2D grid of colors
            grid_size: Size of the grid
            num_colors: Number of colors used
            title: Title for the plot
        """
        plt.figure(figsize=(8, 8))
        
        # Define colors
        cmap = plt.cm.get_cmap('tab10', num_colors)
        
        # Plot grid
        for i in range(grid_size):
            for j in range(grid_size):
                plt.fill([j, j+1, j+1, j], [i, i, i+1, i+1], color=cmap(grid[i][j]))
                plt.text(j+0.5, i+0.5, str(grid[i][j]), ha='center', va='center', fontsize=12)
        
        # Add grid lines
        for i in range(grid_size + 1):
            plt.axhline(i, color='black', linewidth=1)
            plt.axvline(i, color='black', linewidth=1)
        
        plt.xlim(0, grid_size)
        plt.ylim(0, grid_size)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(title or f"{grid_size}x{grid_size} Grid with {num_colors} Colors")
        
        # Create directory for visualizations if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
        
        # Save the plot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"visualizations/grid_{grid_size}x{grid_size}_{num_colors}colors_{timestamp}.png"
        plt.savefig(filename)
        plt.close()
        logger.info(f"Visualization saved to {filename}")

    def generate_and_evaluate(self, grid_size: int, num_colors: int, iterations: int) -> Tuple[bool, List[List[int]], str]:
        """
        Generate and evaluate grid colorings
        
        Args:
            grid_size: Size of the grid
            num_colors: Number of colors to use
            iterations: Number of iterations to run
            
        Returns:
            Tuple of (success, best_grid, best_code)
        """
        successful_examples = []
        
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations}")
            
            # Generate prompt
            prompt = self.generate_prompt(grid_size, num_colors, successful_examples)
            
            # Generate function
            logger.info("Generating function...")
            func_str = self.generate_python_function(prompt)
            
            if not func_str:
                logger.warning("Generated empty function")
                continue
            
            # Run function
            logger.info("Evaluating function...")
            valid, grid = self.run_function(func_str, grid_size, num_colors)
            
            if valid:
                logger.info("Found valid coloring!")
                self.visualize_grid(grid, grid_size, num_colors, f"Valid {grid_size}x{grid_size} Grid with {num_colors} Colors")
                
                # Add to successful examples
                successful_examples.append({
                    'code': func_str,
                    'grid': grid
                })
                
                return True, grid, func_str
            else:
                logger.info("Invalid coloring, continuing search...")
        
        logger.info(f"No valid coloring found after {iterations} iterations")
        return False, [], ""


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="FunSearch implementation for L-shape Ramsey problem")
    parser.add_argument("--grid_size", type=int, default=4, help="Size of the grid")
    parser.add_argument("--colors", type=int, default=3, help="Number of colors")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for generation")
    args = parser.parse_args()
    
    # Initialize FunSearch
    funsearch = TinyLlamaFunSearch(
        model_path=args.model_path,
        temperature=args.temperature
    )
    
    # Run FunSearch
    start_time = time.time()
    success, grid, code = funsearch.generate_and_evaluate(
        grid_size=args.grid_size,
        num_colors=args.colors,
        iterations=args.iterations
    )
    end_time = time.time()
    
    # Report results
    if success:
        logger.info(f"Success! Found valid {args.grid_size}x{args.grid_size} grid with {args.colors} colors")
        logger.info(f"Grid: {grid}")
        logger.info(f"Code: {code}")
    else:
        logger.info(f"No valid solution found for {args.grid_size}x{args.grid_size} grid with {args.colors} colors")
    
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
