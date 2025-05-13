#!/usr/bin/env python3

import numpy as np
from l_shape_ramsey import LShapeGrid, Color
import os
import random

def create_optimized_grid(size):
    """Create a grid with a block pattern to avoid L-shapes."""
    grid = LShapeGrid(size)
    
    # Use a block-based pattern with careful distribution of colors
    for y in range(size):
        for x in range(size):
            # Use a block size of 2
            block_x = x // 2
            block_y = y // 2
            
            # Determine color based on position in block and block position
            if (block_x + block_y) % 2 == 0:
                # Even blocks use one pattern
                color_idx = (x + y) % 3
            else:
                # Odd blocks use a different pattern
                color_idx = (2*x + y) % 3
                
            grid.set_color(x, y, list(Color)[color_idx])
    
    return grid

def check_grid_for_l_shapes(grid):
    """Check if a grid has any L-shapes and visualize it."""
    print(f"Checking {grid.size}×{grid.size} grid for L-shapes...")
    
    # Check for L-shapes
    has_l, points = grid.has_any_l_shape()
    
    if has_l:
        print(f"❌ Grid has an L-shape at points: {points}")
    else:
        print(f"✅ Grid is L-shape free!")
    
    # Save visualization
    filename = f"grid_{grid.size}x{grid.size}_solution.png"
    grid.visualize(highlight_l_shape=has_l, filename=filename)
    print(f"Grid visualization saved to {filename}")
    
    return not has_l

def main():
    # Test for specific grid sizes 
    sizes = [3, 4, 5]
    
    for size in sizes:
        print(f"\n{'='*30}")
        print(f"Testing grid size {size}×{size}")
        print(f"{'='*30}")
        
        grid = create_optimized_grid(size)
        check_grid_for_l_shapes(grid)

if __name__ == "__main__":
    main()