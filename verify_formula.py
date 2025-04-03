#!/usr/bin/env python3
import numpy as np
from l_shape_ramsey import LShapeGrid, Color

def generate_grid(n, formula='mod3mod2'):
    """Generate a grid using different formulas."""
    grid = np.zeros((n, n), dtype=int)
    
    if formula == 'mod3mod2':
        # The formula that should work for many grid sizes
        for i in range(n):
            for j in range(n):
                grid[i, j] = (i + 2*j) % 3 % 2
    elif formula == 'mod3':
        # Simple modular arithmetic
        for i in range(n):
            for j in range(n):
                grid[i, j] = (i + 2*j) % 3
    elif formula == 'alternating':
        # Simple alternating pattern
        for i in range(n):
            for j in range(n):
                grid[i, j] = (i + j) % 2
    elif formula == 'block':
        # Block pattern
        for i in range(n):
            for j in range(n):
                block_i = i // 2
                block_j = j // 2
                grid[i, j] = (block_i + block_j) % 2
    elif formula == 'latin_square':
        # Latin square pattern for 3x3
        pattern = [
            [0, 1, 2],
            [2, 0, 1],
            [1, 2, 0]
        ]
        for i in range(min(n, 3)):
            for j in range(min(n, 3)):
                grid[i, j] = pattern[i][j]
    
    # Special patterns from deterministic_solver.py
    elif formula == '4x4_special':
        if n >= 4:
            pattern = [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0]
            ]
            for i in range(min(n, 4)):
                for j in range(min(n, 4)):
                    grid[i, j] = pattern[i][j]
    
    elif formula == '5x5_special':
        if n >= 5:
            pattern = [
                [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0]
            ]
            for i in range(min(n, 5)):
                for j in range(min(n, 5)):
                    grid[i, j] = pattern[i][j]
    
    elif formula == '6x6_special':
        if n >= 6:
            pattern = [
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [1, 1, 0, 0, 1, 1],
                [1, 1, 0, 0, 1, 1],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0]
            ]
            for i in range(min(n, 6)):
                for j in range(min(n, 6)):
                    grid[i, j] = pattern[i][j]
    
    elif formula == '7x7_special':
        if n >= 7:
            pattern = [
                [0, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [1, 0, 0, 1, 0, 0, 1],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 1, 0, 0, 0]
            ]
            for i in range(min(n, 7)):
                for j in range(min(n, 7)):
                    grid[i, j] = pattern[i][j]
    
    return grid

def verify_grid(grid):
    """Check if a grid has any monochromatic L-shapes."""
    n = grid.shape[0]
    
    # Convert to LShapeGrid
    l_shape_grid = LShapeGrid(n)
    for i in range(n):
        for j in range(n):
            l_shape_grid.set_color(j, i, list(Color)[grid[i, j]])
    
    # Check for L-shapes
    has_l, points = l_shape_grid.has_any_l_shape()
    
    return not has_l, points

def main():
    # Test different grid sizes
    grid_sizes = [3, 4, 5, 6, 7, 8]
    formulas = [
        'mod3mod2',
        'mod3',
        'alternating',
        'block',
        'latin_square',
        '4x4_special',
        '5x5_special',
        '6x6_special',
        '7x7_special'
    ]
    
    print("Testing formulas for L-shape Ramsey problem:")
    print("=" * 50)
    
    for size in grid_sizes:
        print(f"\nGrid size {size}×{size}:")
        print("-" * 20)
        
        for formula in formulas:
            grid = generate_grid(size, formula)
            is_valid, points = verify_grid(grid)
            
            status = "VALID" if is_valid else f"INVALID (L-shape at {points})"
            print(f"Formula '{formula}': {status}")
            
            # If valid, also print the grid for reference
            if is_valid:
                print("Valid grid:")
                print(grid)

if __name__ == "__main__":
    main() 