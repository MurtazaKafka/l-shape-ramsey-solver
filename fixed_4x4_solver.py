#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

# Create a specialized pattern for a 4×4 grid
def generate_4x4_solution():
    """Generate a valid solution for a 4×4 grid."""
    pattern = np.array([
        [0, 2, 0, 2],
        [1, 0, 1, 0],
        [2, 1, 2, 1],
        [0, 2, 0, 2]
    ])
    return pattern

# Verify the solution
def verify_solution(grid):
    """Verify if the grid contains any monochromatic L-shapes."""
    n = len(grid)
    
    for i in range(n):
        for j in range(n):
            color = grid[i, j]
            
            # Check all possible L-shapes starting from this point
            for d in range(1, n):
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

# Visualize the solution
def visualize_solution(grid):
    """Visualize the grid solution."""
    plt.figure(figsize=(8, 8))
    
    # Use three colors: red, green, blue
    cmap = plt.cm.colors.ListedColormap(['red', 'green', 'blue'])
    
    plt.imshow(grid, cmap=cmap)
    plt.grid(True, color='black', linewidth=1.5)
    plt.xticks(range(len(grid)))
    plt.yticks(range(len(grid)))
    plt.title("L-shape Ramsey 4×4 Grid Solution")
    
    # Save the visualization
    os.makedirs("visualizations", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"visualizations/specialized_4x4_{timestamp}.png"
    plt.savefig(filename)
    plt.close()
    
    print(f"Visualization saved to {filename}")

# Main function
def main():
    print("Generating 4×4 solution...")
    grid = generate_4x4_solution()
    
    print("Solution:")
    print(grid)
    
    print("\nVerifying solution...")
    is_valid = verify_solution(grid)
    
    if is_valid:
        print("SUCCESS: The solution is valid (no monochromatic L-shapes).")
        visualize_solution(grid)
    else:
        print("FAILED: The solution contains monochromatic L-shapes.")

if __name__ == "__main__":
    main() 