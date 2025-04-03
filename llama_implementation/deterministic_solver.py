#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, List, Tuple
import time

class DeterministicSolver:
    """Deterministic solver for the L-shape Ramsey problem without using LLMs."""
    
    def __init__(self):
        """Initialize the solver."""
        self.solutions = {}  # Cache for solutions
    
    def verify_solution(self, grid: np.ndarray) -> bool:
        """Verify if a given grid coloring is valid (no monochromatic L-shapes)."""
        n = len(grid)
        
        # Check all possible L-shapes
        for i in range(n-2):
            for j in range(n-2):  # Changed to n-2 to avoid index out of bounds
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
    
    def create_modular_pattern(self, grid_size: int, modulo: int = 3) -> np.ndarray:
        """Create a grid coloring using modular arithmetic pattern.
        
        For 3×3 grids, a (x + 2y) mod 3 pattern works well.
        """
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        for i in range(grid_size):
            for j in range(grid_size):
                grid[i, j] = (i + 2*j) % modulo % 2
                
        return grid
    
    def create_alternating_pattern(self, grid_size: int) -> np.ndarray:
        """Create a grid coloring using alternating rows and columns pattern."""
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        for i in range(grid_size):
            for j in range(grid_size):
                grid[i, j] = (i + j) % 2
                
        return grid
    
    def create_block_pattern(self, grid_size: int) -> np.ndarray:
        """Create a grid coloring using 2×2 blocks with alternating colors."""
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        for i in range(grid_size):
            for j in range(grid_size):
                block_i = i // 2
                block_j = j // 2
                grid[i, j] = (block_i + block_j) % 2
                
        return grid
    
    def create_recursive_pattern(self, grid_size: int) -> np.ndarray:
        """Create a recursive pattern for larger grids."""
        # Base case: 3×3 grid with known solution
        if grid_size <= 3:
            return self.create_modular_pattern(grid_size)
            
        # For 4×4, use block pattern
        if grid_size == 4:
            return np.array([
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0]
            ])
            
        # For 5×5, use modified modular pattern
        if grid_size == 5:
            return np.array([
                [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0]
            ])
            
        # For 6×6, use block pattern with modification
        if grid_size == 6:
            return np.array([
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [1, 1, 0, 0, 1, 1],
                [1, 1, 0, 0, 1, 1],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0]
            ])
            
        # For 7×7, use custom pattern that works
        if grid_size == 7:
            return np.array([
                [0, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [1, 0, 0, 1, 0, 0, 1],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 1, 0, 0, 0]
            ])
        
        # For larger sizes, use a divide-and-conquer approach
        # This is a simplified approach and may not work for all grid sizes
        half_size = grid_size // 2
        remainder = grid_size % 2
        
        # Create a smaller grid and expand it
        smaller_grid = self.create_recursive_pattern(half_size)
        
        # Create the larger grid by duplicating and flipping
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Fill the four quadrants
        grid[:half_size, :half_size] = smaller_grid
        grid[:half_size, half_size:half_size*2] = 1 - smaller_grid
        grid[half_size:half_size*2, :half_size] = 1 - smaller_grid
        grid[half_size:half_size*2, half_size:half_size*2] = smaller_grid
        
        # Handle odd sizes by filling the remaining row/column
        if remainder == 1:
            for i in range(grid_size-1):
                grid[i, grid_size-1] = grid[i, grid_size-2]
                grid[grid_size-1, i] = grid[grid_size-2, i]
            grid[grid_size-1, grid_size-1] = grid[0, 0]
            
        return grid
    
    def solve(self, grid_size: int) -> Optional[np.ndarray]:
        """Solve the L-shape Ramsey problem for the given grid size."""
        
        print(f"Solving {grid_size}×{grid_size} grid...")
        
        # Check if solution is already cached
        if grid_size in self.solutions:
            print(f"Using cached solution for {grid_size}×{grid_size}")
            return self.solutions[grid_size]
            
        # Try different patterns
        patterns = [
            ("Modular", lambda: self.create_modular_pattern(grid_size)),
            ("Alternating", lambda: self.create_alternating_pattern(grid_size)),
            ("Block", lambda: self.create_block_pattern(grid_size)),
            ("Recursive", lambda: self.create_recursive_pattern(grid_size))
        ]
        
        for pattern_name, pattern_func in patterns:
            print(f"Trying {pattern_name} pattern...")
            grid = pattern_func()
            
            if self.verify_solution(grid):
                print(f"Found valid solution using {pattern_name} pattern!")
                self.solutions[grid_size] = grid
                self.visualize_solution(grid, grid_size, pattern_name)
                return grid
        
        print(f"No valid solution found for {grid_size}×{grid_size} grid")
        return None
    
    def visualize_solution(self, grid: np.ndarray, grid_size: int, pattern_name: str = ""):
        """Visualize the solution grid."""
        plt.figure(figsize=(8, 8))
        cmap = plt.cm.colors.ListedColormap(['red', 'blue'])
        plt.imshow(grid, cmap=cmap)
        plt.grid(True, color='black', linewidth=1.5)
        plt.xticks(range(grid_size))
        plt.yticks(range(grid_size))
        title = f"L-shape Ramsey {grid_size}×{grid_size} Grid Solution"
        if pattern_name:
            title += f" ({pattern_name} Pattern)"
        plt.title(title)
        
        # Save the visualization
        os.makedirs("visualizations", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"visualizations/l_shape_grid_{grid_size}x{grid_size}_{timestamp}.png")
        plt.close()
        
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Solve L-shape Ramsey problem using deterministic methods')
    parser.add_argument('--grid-sizes', type=int, nargs='+', default=[3, 4, 5, 6, 7, 8],
                      help='Grid sizes to solve (default: 3 4 5 6 7 8)')
    args = parser.parse_args()
    
    # Initialize solver
    solver = DeterministicSolver()
    
    # Test on different grid sizes
    grid_sizes = args.grid_sizes
    
    results = {}
    
    for size in grid_sizes:
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