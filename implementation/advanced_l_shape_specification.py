"""Advanced L-shape Ramsey Problem Specification for FunSearch.

This specification defines the L-shape Ramsey problem for FunSearch to evolve solutions for.
The problem asks: what is the largest n×n grid that can be colored with k colors
without forming a monochromatic L-shape?
"""
import numpy as np
import torch
import random
import math
from enum import Enum
from typing import List, Tuple, Dict, Set, Optional, Any

# Use FunSearch decorators
import funsearch

class Color(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2

def get_device():
    """Get the appropriate device for PyTorch operations"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class LShapeRamseyGrid:
    """L-shape Ramsey Grid implementation with GPU acceleration."""
    
    def __init__(self, size: int, num_colors: int = 3):
        self.size = size
        self.num_colors = num_colors
        self.device = get_device()
        # Initialize grid with -1 (empty cells)
        self.grid = torch.full((size, size), -1, dtype=torch.int32, device=self.device)
        
    def has_l_shape(self) -> bool:
        """Check if the grid contains any monochromatic L-shape."""
        for c in range(self.num_colors):
            # Create a binary mask where cells of color c are 1, others are 0
            color_mask = (self.grid == c).to(torch.int32)
            
            # Check all possible L-shapes
            for y in range(self.size):
                for x in range(self.size):
                    if color_mask[y, x] == 0:
                        continue
                        
                    for d in range(1, self.size):
                        # Right and Up
                        if (x + d < self.size and y + d < self.size and
                            color_mask[y, x + d] == 1 and 
                            color_mask[y + d, x + d] == 1):
                            return True
                            
                        # Right and Down
                        if (x + d < self.size and y - d >= 0 and
                            color_mask[y, x + d] == 1 and 
                            color_mask[y - d, x + d] == 1):
                            return True
                            
                        # Left and Up
                        if (x - d >= 0 and y + d < self.size and
                            color_mask[y, x - d] == 1 and
                            color_mask[y + d, x - d] == 1):
                            return True
                            
                        # Left and Down
                        if (x - d >= 0 and y - d >= 0 and
                            color_mask[y, x - d] == 1 and
                            color_mask[y - d, x - d] == 1):
                            return True
        
        return False
    
    def compute_color_diversity(self) -> float:
        """Compute a measure of color diversity in the grid."""
        diversity = 0.0
        
        # Check row diversity
        for y in range(self.size):
            row_colors = set()
            for x in range(self.size):
                if self.grid[y, x] >= 0:
                    row_colors.add(self.grid[y, x].item())
            diversity += len(row_colors)
        
        # Check column diversity
        for x in range(self.size):
            col_colors = set()
            for y in range(self.size):
                if self.grid[y, x] >= 0:
                    col_colors.add(self.grid[y, x].item())
            diversity += len(col_colors)
        
        # Normalize
        max_diversity = 2 * self.size * min(self.size, self.num_colors)
        return diversity / max_diversity
    
    def compute_pattern_complexity(self) -> float:
        """Compute a measure of pattern complexity (more complex patterns are better)."""
        complexity = 0.0
        
        # Count local color transitions
        for y in range(self.size - 1):
            for x in range(self.size - 1):
                center = self.grid[y, x].item()
                right = self.grid[y, x + 1].item()
                down = self.grid[y + 1, x].item()
                diag = self.grid[y + 1, x + 1].item()
                
                # Count different transitions
                transitions = len(set([center, right, down, diag]))
                complexity += transitions - 1  # -1 because we want to count transitions, not colors
        
        # Normalize
        max_complexity = (self.size - 1) * (self.size - 1) * (min(4, self.num_colors) - 1)
        if max_complexity == 0:
            return 0.0
        return complexity / max_complexity
    
    def fill_grid_from_genotype(self, genotype: Dict[str, Any]) -> None:
        """Fill the grid based on a genotype specification."""
        # Reset grid
        self.grid = torch.full((self.size, self.size), -1, dtype=torch.int32, device=self.device)
        
        # Extract parameters
        pattern_type = genotype.get("pattern_type", "modulo")
        params = genotype.get("params", {})
        
        if pattern_type == "modulo":
            # Modular arithmetic pattern: (ax + by + c) % num_colors
            a = params.get("a", 1)
            b = params.get("b", 2)
            c = params.get("c", 0)
            
            for y in range(self.size):
                for x in range(self.size):
                    color = (a*x + b*y + c) % self.num_colors
                    self.grid[y, x] = color
                    
        elif pattern_type == "block":
            # Block-based pattern
            block_size = params.get("block_size", 2)
            
            for block_y in range((self.size + block_size - 1) // block_size):
                for block_x in range((self.size + block_size - 1) // block_size):
                    pattern_idx = (block_x + block_y) % self.num_colors
                    
                    for dy in range(block_size):
                        for dx in range(block_size):
                            y = block_y * block_size + dy
                            x = block_x * block_size + dx
                            
                            if x < self.size and y < self.size:
                                if pattern_idx == 0:
                                    # Diagonal pattern within block
                                    color = (dx + dy) % self.num_colors
                                elif pattern_idx == 1:
                                    # Reverse diagonal within block
                                    color = (dx + (block_size - 1 - dy)) % self.num_colors
                                else:
                                    # Other patterns
                                    color = (dx * dy) % self.num_colors
                                
                                self.grid[y, x] = color
        
        elif pattern_type == "recursive":
            # Recursive subdivision pattern
            self._fill_recursive(0, 0, self.size, params)
            
        elif pattern_type == "formula":
            # Use mathematical formula
            formula_type = params.get("formula_type", 0)
            
            if formula_type == 0:
                # Formula: (x^2 + y^2) % num_colors
                for y in range(self.size):
                    for x in range(self.size):
                        color = (x*x + y*y) % self.num_colors
                        self.grid[y, x] = color
            
            elif formula_type == 1:
                # Formula: (x*y) % num_colors
                for y in range(self.size):
                    for x in range(self.size):
                        color = (x*y) % self.num_colors
                        self.grid[y, x] = color
            
            elif formula_type == 2:
                # Formula based on golden ratio
                phi = (1 + math.sqrt(5)) / 2
                for y in range(self.size):
                    for x in range(self.size):
                        z = x * phi + y / phi
                        color = int(self.num_colors * (z - int(z)))
                        self.grid[y, x] = color
            
            else:
                # Default formula
                for y in range(self.size):
                    for x in range(self.size):
                        color = (x + 2*y) % self.num_colors
                        self.grid[y, x] = color
                        
        elif pattern_type == "random":
            # Random pattern with constraints
            for y in range(self.size):
                for x in range(self.size):
                    # Check neighbors to avoid L-shapes
                    used_colors = set()
                    
                    # Check potential L-shapes
                    for d in range(1, min(self.size, 4)):  # Limit the search distance for efficiency
                        # Check right L-shapes
                        if x + d < self.size:
                            if y + d < self.size and self.grid[y, x + d] >= 0 and self.grid[y + d, x + d] >= 0:
                                if self.grid[y, x + d].item() == self.grid[y + d, x + d].item():
                                    used_colors.add(self.grid[y, x + d].item())
                            
                            if y - d >= 0 and self.grid[y, x + d] >= 0 and self.grid[y - d, x + d] >= 0:
                                if self.grid[y, x + d].item() == self.grid[y - d, x + d].item():
                                    used_colors.add(self.grid[y, x + d].item())
                        
                        # Check left L-shapes
                        if x - d >= 0:
                            if y + d < self.size and self.grid[y, x - d] >= 0 and self.grid[y + d, x - d] >= 0:
                                if self.grid[y, x - d].item() == self.grid[y + d, x - d].item():
                                    used_colors.add(self.grid[y, x - d].item())
                            
                            if y - d >= 0 and self.grid[y, x - d] >= 0 and self.grid[y - d, x - d] >= 0:
                                if self.grid[y, x - d].item() == self.grid[y - d, x - d].item():
                                    used_colors.add(self.grid[y, x - d].item())
                    
                    # Choose a color not in used_colors if possible
                    available_colors = [c for c in range(self.num_colors) if c not in used_colors]
                    if available_colors:
                        color = random.choice(available_colors)
                    else:
                        # If all colors would create an L-shape, choose randomly
                        color = random.randint(0, self.num_colors - 1)
                    
                    self.grid[y, x] = color
        
        else:
            # Default pattern if unknown type
            for y in range(self.size):
                for x in range(self.size):
                    color = (x + 2*y) % self.num_colors
                    self.grid[y, x] = color
    
    def _fill_recursive(self, start_x: int, start_y: int, size: int, params: Dict[str, Any]) -> None:
        """Recursively fill a section of the grid."""
        if size <= 1:
            # Base case: single cell
            if start_x < self.size and start_y < self.size:
                self.grid[start_y, start_x] = random.randint(0, self.num_colors - 1)
            return
            
        # For known small cases, use optimal patterns
        if size <= 3 and start_x + size <= self.size and start_y + size <= self.size:
            if size == 2:
                # 2x2 pattern (alternating)
                self.grid[start_y, start_x] = 0
                self.grid[start_y, start_x + 1] = 1
                self.grid[start_y + 1, start_x] = 1
                self.grid[start_y + 1, start_x + 1] = 0
                return
            elif size == 3:
                # 3x3 Latin square (known to work)
                pattern = [
                    [0, 1, 2],
                    [2, 0, 1],
                    [1, 2, 0]
                ]
                for y in range(3):
                    for x in range(3):
                        if start_y + y < self.size and start_x + x < self.size:
                            self.grid[start_y + y, start_x + x] = pattern[y][x]
                return
        
        # Otherwise, divide and conquer
        half = size // 2
        extra = size % 2
        
        # Recursively fill quadrants with different patterns
        self._fill_recursive(start_x, start_y, half + extra, params)
        self._fill_recursive(start_x + half + extra, start_y, half, params)
        self._fill_recursive(start_x, start_y + half + extra, half, params)
        self._fill_recursive(start_x + half + extra, start_y + half + extra, half, params)
        
        # After recursive filling, fix boundaries to avoid L-shapes
        if size > 3:
            self._fix_boundaries(start_x, start_y, size, params)
    
    def _fix_boundaries(self, start_x: int, start_y: int, size: int, params: Dict[str, Any]) -> None:
        """Fix the boundaries between quadrants to avoid L-shapes."""
        half = size // 2
        extra = size % 2
        mid_x = start_x + half + extra - 1
        mid_y = start_y + half + extra - 1
        
        # Check and fix horizontal boundary
        for x in range(start_x, start_x + size):
            if x < self.size and mid_y < self.size and mid_y + 1 < self.size:
                # Check if this creates an L-shape
                for d in range(1, 4):  # Check a few nearby cells
                    if (x + d < self.size and 
                        self.grid[mid_y, x].item() == self.grid[mid_y, x + d].item() == self.grid[mid_y + 1, x + d].item()):
                        # Fix by changing the color
                        new_color = (self.grid[mid_y, x].item() + 1) % self.num_colors
                        self.grid[mid_y + 1, x + d] = new_color
                        
                    if (x - d >= 0 and 
                        self.grid[mid_y, x].item() == self.grid[mid_y, x - d].item() == self.grid[mid_y + 1, x - d].item()):
                        # Fix by changing the color
                        new_color = (self.grid[mid_y, x].item() + 1) % self.num_colors
                        self.grid[mid_y + 1, x - d] = new_color
        
        # Check and fix vertical boundary
        for y in range(start_y, start_y + size):
            if y < self.size and mid_x < self.size and mid_x + 1 < self.size:
                # Check if this creates an L-shape
                for d in range(1, 4):  # Check a few nearby cells
                    if (y + d < self.size and 
                        self.grid[y, mid_x].item() == self.grid[y + d, mid_x].item() == self.grid[y + d, mid_x + 1].item()):
                        # Fix by changing the color
                        new_color = (self.grid[y, mid_x].item() + 1) % self.num_colors
                        self.grid[y + d, mid_x + 1] = new_color
                        
                    if (y - d >= 0 and 
                        self.grid[y, mid_x].item() == self.grid[y - d, mid_x].item() == self.grid[y - d, mid_x + 1].item()):
                        # Fix by changing the color
                        new_color = (self.grid[y, mid_x].item() + 1) % self.num_colors
                        self.grid[y - d, mid_x + 1] = new_color

# The function that FunSearch will evolve
@funsearch.evolve
def generate_l_shape_ramsey_grid(grid_size: int, num_colors: int = 3, seed: int = 42) -> Dict[str, Any]:
    """
    Generate a grid configuration for the L-shape Ramsey problem.
    
    Args:
        grid_size: Size of the grid (n×n)
        num_colors: Number of colors to use
        seed: Random seed for reproducibility
        
    Returns:
        A dictionary containing the genotype specification for generating the grid
    """
    random.seed(seed)
    
    # Basic genotype template
    genotype = {
        "pattern_type": "modulo",  # Default pattern type
        "params": {
            "a": 1,
            "b": 2,
            "c": 0
        }
    }
    
    # Different strategies based on grid size
    if grid_size <= 3:
        # For small grids, modulo patterns work well
        genotype["pattern_type"] = "modulo"
        genotype["params"]["a"] = 1
        genotype["params"]["b"] = 2
        genotype["params"]["c"] = 0
        
    elif grid_size <= 6:
        # For medium grids, try recursive patterns
        genotype["pattern_type"] = "recursive"
        
    else:
        # For larger grids, try more complex patterns
        pattern_choice = random.randint(0, 3)
        
        if pattern_choice == 0:
            # Modulo pattern with carefully chosen parameters
            genotype["pattern_type"] = "modulo"
            genotype["params"]["a"] = random.choice([1, 2, 3])
            genotype["params"]["b"] = random.choice([1, 2, 3])
            genotype["params"]["c"] = random.randint(0, 5)
            
        elif pattern_choice == 1:
            # Block-based pattern
            genotype["pattern_type"] = "block"
            genotype["params"]["block_size"] = random.choice([2, 3])
            
        elif pattern_choice == 2:
            # Formula-based pattern
            genotype["pattern_type"] = "formula"
            genotype["params"]["formula_type"] = random.randint(0, 3)
            
        else:
            # Recursive pattern
            genotype["pattern_type"] = "recursive"
    
    return genotype

# The function that FunSearch will run and optimize for
@funsearch.run
def evaluate_l_shape_ramsey(problem: Tuple[int, int, int]) -> float:
    """
    Evaluate the quality of the evolved grid generation function.
    
    Args:
        problem: Tuple of (grid_size, num_colors, seed)
        
    Returns:
        A score where higher is better (0 if invalid with L-shapes)
    """
    grid_size, num_colors, seed = problem
    random.seed(seed)
    
    # Create a grid
    grid = LShapeRamseyGrid(grid_size, num_colors)
    
    # Get the genotype from our evolved function
    genotype = generate_l_shape_ramsey_grid(grid_size, num_colors, seed)
    
    # Fill the grid according to the genotype
    grid.fill_grid_from_genotype(genotype)
    
    # Check if the grid has any L-shapes
    has_l_shape = grid.has_l_shape()
    
    if has_l_shape:
        return 0.0  # Invalid grid
    
    # For valid grids, compute a score based on various factors
    
    # Base score is the size of the grid
    base_score = grid_size * grid_size
    
    # Bonus for diversity of colors
    diversity_bonus = grid.compute_color_diversity() * 10.0
    
    # Bonus for pattern complexity
    complexity_bonus = grid.compute_pattern_complexity() * 5.0
    
    # Combine scores
    total_score = base_score + diversity_bonus + complexity_bonus
    
    return total_score 