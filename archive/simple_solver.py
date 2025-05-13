#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
from datetime import datetime

class SimpleLShapeRamseySolver:
    """
    Solver for the L-shape Ramsey problem using simple modular arithmetic.
    
    This implementation uses the formula (i + 2*j) % 3 to create a valid coloring
    that avoids monochromatic L-shapes.
    """
    
    def __init__(self):
        """Initialize the solver."""
        self.visualizations_dir = "visualizations"
        os.makedirs(self.visualizations_dir, exist_ok=True)
    
    def generate_solution(self, grid_size: int) -> np.ndarray:
        """
        Generate a solution for the given grid size using modular arithmetic.
        
        Args:
            grid_size: The size of the grid (n×n)
            
        Returns:
            A numpy array with the grid coloring (values 0, 1, 2)
        """
        print(f"Generating solution for {grid_size}×{grid_size} grid...")
        
        # Create a grid filled with zeros
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Fill the grid using the modular pattern: (i + 2*j) % 3
        # This pattern avoids L-shapes by ensuring adjacent cells have different colors
        for i in range(grid_size):
            for j in range(grid_size):
                grid[i, j] = (i + 2*j) % 3
        
        return grid
    
    def verify_solution(self, grid: np.ndarray) -> bool:
        """
        Verify if a given grid coloring is valid (no monochromatic L-shapes).
        
        An L-shape consists of three points where two points are equidistant from 
        the third point, forming a right angle.
        
        Args:
            grid: The grid to verify
            
        Returns:
            True if valid, False otherwise
        """
        n = len(grid)
        
        # Check all possible L-shapes
        for i in range(n):
            for j in range(n):
                color = grid[i, j]
                
                # Check all possible L-shapes starting from this point
                for d in range(1, n):
                    # Check all four orientations of L-shapes
                    
                    # Right and Up
                    if (j + d) < n and (i + d) < n:
                        if (grid[i, j+d] == color and 
                            grid[i+d, j+d] == color):
                            print(f"Found invalid L-shape (Right+Up) at ({i},{j}) with d={d}")
                            return False
                    
                    # Right and Down
                    if (j + d) < n and (i - d) >= 0:
                        if (grid[i, j+d] == color and 
                            grid[i-d, j+d] == color):
                            print(f"Found invalid L-shape (Right+Down) at ({i},{j}) with d={d}")
                            return False
                    
                    # Left and Up
                    if (j - d) >= 0 and (i + d) < n:
                        if (grid[i, j-d] == color and 
                            grid[i+d, j-d] == color):
                            print(f"Found invalid L-shape (Left+Up) at ({i},{j}) with d={d}")
                            return False
                    
                    # Left and Down
                    if (j - d) >= 0 and (i - d) >= 0:
                        if (grid[i, j-d] == color and 
                            grid[i-d, j-d] == color):
                            print(f"Found invalid L-shape (Left+Down) at ({i},{j}) with d={d}")
                            return False
        
        return True
    
    def visualize_solution(self, grid: np.ndarray, grid_size: int):
        """
        Visualize the solution grid.
        
        Args:
            grid: The grid to visualize
            grid_size: The size of the grid
        """
        plt.figure(figsize=(8, 8))
        
        # Use three colors: red, green, blue
        cmap = plt.cm.colors.ListedColormap(['red', 'green', 'blue'])
            
        plt.imshow(grid, cmap=cmap)
        plt.grid(True, color='black', linewidth=1.5)
        plt.xticks(range(grid_size))
        plt.yticks(range(grid_size))
        plt.title(f"L-shape Ramsey {grid_size}×{grid_size} Grid Solution")
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.visualizations_dir}/simple_grid_{grid_size}x{grid_size}_{timestamp}.png"
        plt.savefig(filename)
        plt.close()
        
        print(f"Visualization saved to {filename}")
    
    def solve(self, grid_sizes: list[int]):
        """
        Solve the L-shape Ramsey problem for multiple grid sizes.
        
        Args:
            grid_sizes: List of grid sizes to solve
        """
        results = {}
        
        for size in grid_sizes:
            start_time = time.time()
            
            # Generate solution
            solution = self.generate_solution(size)
            
            # Verify solution
            is_valid = self.verify_solution(solution)
            
            # Record result
            results[size] = "Solved" if is_valid else "Failed"
            
            # Visualize if valid
            if is_valid:
                self.visualize_solution(solution, size)
            
            end_time = time.time()
            print(f"Time taken for {size}×{size} grid: {end_time - start_time:.2f} seconds")
            print(f"Result: {results[size]}")
            print(f"Solution:\n{solution}")
            print("-" * 50)
        
        # Print summary
        print("\nSummary of results:")
        for size, result in results.items():
            print(f"{size}×{size} grid: {result}")

def main():
    parser = argparse.ArgumentParser(description='Solve L-shape Ramsey problem with modular arithmetic')
    parser.add_argument('--grid-sizes', type=int, nargs='+', default=[3, 4, 5, 6, 7, 8],
                        help='Grid sizes to solve (default: 3 4 5 6 7 8)')
    args = parser.parse_args()
    
    # Initialize solver
    solver = SimpleLShapeRamseySolver()
    
    # Solve for specified grid sizes
    solver.solve(args.grid_sizes)

if __name__ == "__main__":
    main() 