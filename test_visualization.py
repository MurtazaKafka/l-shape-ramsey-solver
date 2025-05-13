#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from l_shape_ramsey import LShapeGrid, Color
import os
import random

def create_latin_square_grid(size):
    """Create a Latin square pattern grid."""
    grid = LShapeGrid(size)
    
    for y in range(size):
        for x in range(size):
            # Standard Latin square pattern
            color_idx = (x + y) % 3
            grid.set_color(x, y, list(Color)[color_idx])
    
    return grid

def create_modular_grid(size):
    """Create a grid based on a more complex modular pattern."""
    grid = LShapeGrid(size)
    
    for y in range(size):
        for x in range(size):
            # More complex formula that avoids many L-shapes
            color_idx = (x + 2*y + (x*y) % 2) % 3
            grid.set_color(x, y, list(Color)[color_idx])
    
    return grid

def create_diagonals_grid(size):
    """Create a grid with diagonal pattern."""
    grid = LShapeGrid(size)
    
    for y in range(size):
        for x in range(size):
            # Based on the main diagonal position
            diagonal_pos = (x + y) % 3
            if diagonal_pos == 0:
                grid.set_color(x, y, Color.RED)
            elif diagonal_pos == 1:
                grid.set_color(x, y, Color.GREEN)
            else:
                grid.set_color(x, y, Color.BLUE)
    
    return grid

def create_checker_grid(size):
    """Create a grid with a checker pattern."""
    grid = LShapeGrid(size)
    
    for y in range(size):
        for x in range(size):
            # Alternating pattern
            if (x % 2 == 0 and y % 2 == 0) or (x % 2 == 1 and y % 2 == 1):
                grid.set_color(x, y, Color.RED)
            elif (x % 2 == 0 and y % 2 == 1):
                grid.set_color(x, y, Color.GREEN)
            else:
                grid.set_color(x, y, Color.BLUE)
    
    return grid

def create_hybrid_grid(size):
    """Create a grid using a hybrid of different patterns."""
    grid = LShapeGrid(size)
    
    for y in range(size):
        for x in range(size):
            # Distance from center (normalized)
            center = size // 2
            dist_from_center = abs(x - center) + abs(y - center)
            
            # Combine multiple factors for color selection
            color_idx = (x + 2*y + dist_from_center) % 3
            grid.set_color(x, y, list(Color)[color_idx])
    
    return grid

def create_optimized_grid(size):
    """Create a grid with a pattern optimized for avoiding L-shapes."""
    grid = LShapeGrid(size)
    
    # Different patterns for different grid sizes
    if size <= 4:
        # Pattern for small grids
        pattern_4x4 = [
            [0, 1, 2, 0],
            [2, 0, 1, 2],
            [1, 2, 0, 1],
            [0, 1, 2, 0]
        ]
        
        for y in range(min(size, 4)):
            for x in range(min(size, 4)):
                grid.set_color(x, y, list(Color)[pattern_4x4[y][x]])
    else:
        # For larger grids, use a block-based approach
        block_size = 3
        
        for y in range(size):
            for x in range(size):
                # Determine which block this cell belongs to
                block_x = x // block_size
                block_y = y // block_size
                
                # Position within the block
                local_x = x % block_size
                local_y = y % block_size
                
                # Use a pattern based on block coordinates and local coordinates
                if (block_x + block_y) % 3 == 0:
                    color_idx = (local_x + local_y) % 3
                elif (block_x + block_y) % 3 == 1:
                    color_idx = (local_x + 2*local_y) % 3
                else:
                    color_idx = (2*local_x + local_y) % 3
                
                grid.set_color(x, y, list(Color)[color_idx])
    
    return grid

def visualize_test_grid(grid, filename=None):
    """Visualize a grid using the built-in visualization."""
    print(f"Visualizing {grid.size}×{grid.size} grid...")
    
    # Create directory if needed
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Visualize with built-in method
    grid.visualize(highlight_l_shape=True, filename=filename)
    
    # Check for L-shapes
    has_l, points = grid.has_any_l_shape()
    print(f"Grid has L-shape: {has_l}")
    if has_l:
        print(f"L-shape found at points: {points}")
    
    return has_l, points

def local_search_optimize(grid, max_iterations=100):
    """Try to optimize a grid using local search."""
    best_grid = grid
    has_l, points = best_grid.has_any_l_shape()
    
    if not has_l:
        print("Grid already optimal (no L-shapes)")
        return grid
    
    print(f"Starting local search optimization, initial L-shape at {points}")
    
    for i in range(max_iterations):
        # Make a copy of the current best grid
        new_grid = LShapeGrid(best_grid.size)
        for y in range(best_grid.size):
            for x in range(best_grid.size):
                new_grid.set_color(x, y, best_grid.get_color(x, y))
        
        # Choose a point from the L-shape or a random point with 50% probability
        if has_l and random.random() < 0.5:
            x, y = random.choice(points)
        else:
            x = random.randint(0, new_grid.size - 1)
            y = random.randint(0, new_grid.size - 1)
        
        # Change the color to a different one
        current_color = new_grid.get_color(x, y)
        new_color = random.choice([c for c in Color if c != current_color])
        new_grid.set_color(x, y, new_color)
        
        # Check if this improves the solution
        new_has_l, new_points = new_grid.has_any_l_shape()
        
        if not new_has_l:
            print(f"Found optimal solution at iteration {i+1}")
            return new_grid
        
        # If still has L-shapes but fewer than before, accept the change
        if len(new_points) < len(points):
            best_grid = new_grid
            has_l = new_has_l
            points = new_points
            print(f"Improved at iteration {i+1}, L-shape at {points}")
    
    print(f"Failed to find optimal solution after {max_iterations} iterations")
    return best_grid

def main():
    # Test various grid sizes
    grid_sizes = [3, 4, 5, 6, 7, 8, 9, 10]
    
    os.makedirs("test_visualizations", exist_ok=True)
    
    for size in grid_sizes:
        print(f"\n{'='*40}")
        print(f"Testing grid size {size}×{size}")
        print(f"{'='*40}")
        
        # Try different grid generation strategies
        print("\nTrying different grid generation strategies:")
        strategies = [
            ("Optimized", create_optimized_grid),
            ("Latin Square", create_latin_square_grid),
            ("Modular", create_modular_grid),
            ("Diagonal", create_diagonals_grid),
            ("Checker", create_checker_grid),
            ("Hybrid", create_hybrid_grid)
        ]
        
        best_grid = None
        best_strategy = None
        
        for name, strategy_func in strategies:
            print(f"\nTrying {name} strategy...")
            grid = strategy_func(size)
            
            filename = f"test_visualizations/grid_{size}x{size}_{name.lower()}.png"
            has_l, points = visualize_test_grid(grid, filename)
            
            if not has_l:
                print(f"✅ Valid solution found with {name} strategy for {size}×{size}")
                best_grid = grid
                best_strategy = name
                break
            else:
                print(f"❌ Invalid solution with {name} strategy for {size}×{size}")
        
        # If no strategy worked directly, try local search
        if best_grid is None:
            print("\nNo direct strategy worked. Trying local search optimization...")
            
            # Start with the optimized grid
            base_grid = create_optimized_grid(size)
            optimized_grid = local_search_optimize(base_grid, max_iterations=1000)
            
            filename = f"test_visualizations/grid_{size}x{size}_optimized_local_search.png"
            has_l, points = visualize_test_grid(optimized_grid, filename)
            
            if not has_l:
                print(f"✅ Valid solution found with local search for {size}×{size}")
                best_grid = optimized_grid
                best_strategy = "Optimized + Local Search"
            else:
                print(f"❌ Failed to find valid solution for {size}×{size}")
        
        # Save the best solution if found
        if best_grid is not None:
            # Save to a standard location for easy reference
            filename = f"grid_{size}x{size}_solution.png"
            best_grid.visualize(highlight_l_shape=False, filename=filename)
            print(f"\n✅ Best solution for {size}×{size} found using {best_strategy} strategy")
            print(f"Solution saved to {filename}")

if __name__ == "__main__":
    main()