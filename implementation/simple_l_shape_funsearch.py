"""Simplified FunSearch implementation for the L-shape Ramsey problem."""
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
from datetime import datetime
from enum import Enum
from typing import List, Tuple, Optional

class Color(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2

class LShapeGrid:
    def __init__(self, size: int):
        self.size = size
        self.grid = np.full((size, size), None, dtype=object)
    
    def set_color(self, x: int, y: int, color: Color) -> None:
        """Set the color at position (x,y)"""
        if 0 <= x < self.size and 0 <= y < self.size:
            self.grid[y][x] = color
    
    def get_color(self, x: int, y: int) -> Optional[Color]:
        """Get the color at position (x,y)"""
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.grid[y][x]
        return None
    
    def has_l_shape(self, x: int, y: int) -> Tuple[bool, List[Tuple[int, int]]]:
        """Check if there's a monochromatic L-shape starting at position (x,y)"""
        color = self.get_color(x, y)
        if color is None:
            return False, []
            
        # Check all possible L-shapes starting from this point
        for d in range(1, self.size):
            # Check all four orientations of L-shapes
            
            # Right and Up
            if (x + d) < self.size and (y + d) < self.size:
                if (self.get_color(x + d, y) == color and 
                    self.get_color(x + d, y + d) == color):
                    return True, [(x, y), (x + d, y), (x + d, y + d)]
            
            # Right and Down
            if (x + d) < self.size and (y - d) >= 0:
                if (self.get_color(x + d, y) == color and 
                    self.get_color(x + d, y - d) == color):
                    return True, [(x, y), (x + d, y), (x + d, y - d)]
            
            # Left and Up
            if (x - d) >= 0 and (y + d) < self.size:
                if (self.get_color(x - d, y) == color and 
                    self.get_color(x - d, y + d) == color):
                    return True, [(x, y), (x - d, y), (x - d, y + d)]
            
            # Left and Down
            if (x - d) >= 0 and (y - d) >= 0:
                if (self.get_color(x - d, y) == color and 
                    self.get_color(x - d, y - d) == color):
                    return True, [(x, y), (x - d, y), (x - d, y - d)]
                    
        return False, []
    
    def has_any_l_shape(self) -> Tuple[bool, List[Tuple[int, int]]]:
        """Check if the grid contains any monochromatic L-shape"""
        for x in range(self.size):
            for y in range(self.size):
                has_l, points = self.has_l_shape(x, y)
                if has_l:
                    return True, points
        return False, []
    
    def visualize(self, filename=None):
        """Visualize the grid with matplotlib"""
        fig, ax = plt.subplots(figsize=(8, 8))
        color_map = {
            Color.RED: 'red',
            Color.GREEN: 'green',
            Color.BLUE: 'blue',
            None: 'white'
        }

        # Draw the grid
        for y in range(self.size):
            for x in range(self.size):
                color = self.get_color(x, y)
                cell_color = color_map[color]
                rect = plt.Rectangle((x, self.size - 1 - y), 1, 1, 
                                     facecolor=cell_color, edgecolor='black')
                ax.add_patch(rect)
        
        # Set plot limits and grid
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_xticks(range(self.size + 1))
        ax.set_yticks(range(self.size + 1))
        ax.grid(True)
        
        # Add title with size info
        plt.title(f'L-shape Ramsey Grid ({self.size}x{self.size})')
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def generate_l_shape_grid(size: int, seed: int) -> Tuple[LShapeGrid, float, bool, List[Tuple[int, int]]]:
    """Generate a grid with the given size and seed"""
    random.seed(seed)
    grid = LShapeGrid(size)
    
    # Choose strategy based on seed
    strategy = seed % 8
    
    # Strategy 1: Mathematically-derived pattern avoiding L-shapes
    if strategy == 0:
        for y in range(size):
            for x in range(size):
                # This formula is designed to avoid creating L-shapes
                color_idx = (x + 2*y) % 3
                grid.set_color(x, y, list(Color)[color_idx])
    
    # Strategy 2: Diagonal stripe pattern
    elif strategy == 1:
        for y in range(size):
            for x in range(size):
                color_idx = (x + y) % 3
                grid.set_color(x, y, list(Color)[color_idx])
    
    # Strategy 3: Modified block pattern with varied block sizes
    elif strategy == 2:
        block_size = max(2, size // 3)
        for y in range(size):
            for x in range(size):
                block_y = y // block_size
                block_x = x // block_size
                color_idx = (block_y + 2*block_x) % 3
                grid.set_color(x, y, list(Color)[color_idx])
    
    # Strategy 4: Triple modulo pattern
    elif strategy == 3:
        for y in range(size):
            for x in range(size):
                color_idx = (x % 3 + y % 3) % 3
                grid.set_color(x, y, list(Color)[color_idx])
    
    # Strategy 5: Knight's move pattern
    elif strategy == 4:
        for y in range(size):
            for x in range(size):
                # Pattern based on knight's move in chess
                color_idx = ((2*x + y) % 3)
                grid.set_color(x, y, list(Color)[color_idx])
    
    # Strategy 6: Quadrant-based coloring
    elif strategy == 5:
        mid_x = size // 2
        mid_y = size // 2
        for y in range(size):
            for x in range(size):
                # Determine quadrant (0-3)
                quadrant = (1 if x >= mid_x else 0) + (2 if y >= mid_y else 0)
                # Use different patterns for each quadrant
                if quadrant == 0:
                    color_idx = (x + y) % 3
                elif quadrant == 1:
                    color_idx = (x - y + size) % 3
                elif quadrant == 2:
                    color_idx = (y - x + size) % 3
                else:  # quadrant == 3
                    color_idx = (2*x + y) % 3
                grid.set_color(x, y, list(Color)[color_idx])
    
    # Strategy 7: Triangular pattern
    elif strategy == 6:
        for y in range(size):
            for x in range(size):
                color_idx = (x + y + (x*y) % 3) % 3
                grid.set_color(x, y, list(Color)[color_idx])
    
    # Strategy 8: Prime-based pattern
    else:
        # Simple list of first few primes
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for y in range(size):
            for x in range(size):
                # Use modular arithmetic with primes to create patterns
                prime_idx = (x + y) % len(primes)
                color_idx = (x * y + primes[prime_idx]) % 3
                grid.set_color(x, y, list(Color)[color_idx])
    
    # Check for L-shapes
    has_l, l_points = grid.has_any_l_shape()
    
    # Calculate score
    if has_l:
        # Score of 0 for invalid grids
        score = 0
    else:
        # Valid grid: higher score for larger grids
        score = size * size
        
        # Bonus for interesting patterns (more than one color)
        unique_colors = set()
        for y in range(size):
            for x in range(size):
                unique_colors.add(grid.get_color(x, y))
        
        score += len(unique_colors) * 5
    
    return grid, score, has_l, l_points


def evolve_grid(grid_description: Tuple[int, int]) -> Tuple[int, int]:
    """Evolve a grid configuration by modifying the seed"""
    size, seed = grid_description
    
    # Evolution strategies
    strategy = random.random()
    
    # 60% chance: Small mutation to explore similar solutions
    if strategy < 0.6:
        new_seed = seed + random.randint(-5, 5)
    
    # 20% chance: Medium mutation
    elif strategy < 0.8:
        new_seed = seed + random.randint(-100, 100)
    
    # 15% chance: Completely new seed
    elif strategy < 0.95:
        new_seed = random.randint(1, 10000)
    
    # 5% chance: Use a seed based on the current time for diversity
    else:
        new_seed = int(time.time() * 1000) % 10000
    
    return (size, new_seed)


def simple_funsearch():
    """Run a simplified version of FunSearch for the L-shape Ramsey problem"""
    # Create visualizations directory
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Grid sizes to try
    grid_sizes = [3, 4, 5, 6, 7, 8]
    
    # Number of iterations per grid size
    iterations = 1000
    
    # Best results for each size
    best_results = {size: {"score": 0, "seed": None, "grid": None} for size in grid_sizes}
    
    print(f"Starting Simple FunSearch for L-shape Ramsey Problem")
    print(f"Grid sizes: {grid_sizes}")
    print(f"Iterations per size: {iterations}")
    print("=" * 50)
    
    start_time = time.time()
    
    for size in grid_sizes:
        print(f"\nSearching for grid size {size}x{size}")
        
        # Initial seeds - try multiple starting points
        seeds = [42, 100, 500, 1000, 5000]
        
        # For larger grid sizes, use more iterations
        size_iterations = iterations * (1 + (size - 3) // 2)
        iterations_per_seed = size_iterations // len(seeds)
        
        total_iterations = 0
        for initial_seed in seeds:
            print(f"  Starting with seed {initial_seed}")
            grid_description = (size, initial_seed)
            
            found_valid = False
            for i in range(iterations_per_seed):
                total_iterations += 1
                
                # Generate grid
                grid, score, has_l, l_points = generate_l_shape_grid(size, grid_description[1])
                
                # Update best result
                if score > best_results[size]["score"]:
                    best_results[size]["score"] = score
                    best_results[size]["seed"] = grid_description[1]
                    best_results[size]["grid"] = grid
                    
                    print(f"    Iteration {total_iterations}/{size_iterations}: New best score {score} (seed {grid_description[1]})")
                    
                    # Save visualization
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{viz_dir}/l_shape_grid_{size}x{size}_{timestamp}.png"
                    grid.visualize(filename)
                    
                    # If we found a valid grid (no L-shapes), we can move to next grid size
                    if not has_l:
                        found_valid = True
                
                # Report progress periodically
                if (i+1) % 100 == 0:
                    print(f"    Completed {total_iterations}/{size_iterations} iterations, current best score: {best_results[size]['score']}")
                
                # Evolve grid description
                grid_description = evolve_grid(grid_description)
                
                # Stop early if we've found a valid solution
                if found_valid and i > iterations_per_seed // 4:
                    break
            
            # If we've found a valid solution, we can move to the next grid size
            if found_valid:
                print(f"  Found valid solution for {size}x{size}, moving to next grid size")
                break
        
        # Print best result for this size
        print(f"Best result for {size}x{size}: Score {best_results[size]['score']} (seed {best_results[size]['seed']})")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Summary of best results:")
    for size in grid_sizes:
        result = best_results[size]
        print(f"Grid {size}x{size}: Score {result['score']} (seed {result['seed']})")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f} seconds")


if __name__ == "__main__":
    simple_funsearch() 