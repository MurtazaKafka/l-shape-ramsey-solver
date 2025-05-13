"""
Advanced L-shape Ramsey Problem Solver

This module implements specialized strategies for solving the L-shape Ramsey problem,
which asks for the largest grid that can be colored using k colors without creating
any monochromatic L-shapes.
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
from datetime import datetime
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any, Set


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
    
    def clone(self):
        """Create a deep copy of the grid"""
        new_grid = LShapeGrid(self.size)
        for y in range(self.size):
            for x in range(self.size):
                new_grid.grid[y][x] = self.grid[y][x]
        return new_grid
    
    def to_array(self):
        """Convert the grid to a 2D numpy array of integers"""
        array = np.zeros((self.size, self.size), dtype=int)
        for y in range(self.size):
            for x in range(self.size):
                color = self.grid[y][x]
                if color is not None:
                    array[y, x] = color.value + 1
        return array
    
    def from_array(self, array):
        """Load the grid from a 2D numpy array of integers"""
        assert array.shape == (self.size, self.size)
        for y in range(self.size):
            for x in range(self.size):
                if array[y, x] > 0:
                    self.grid[y][x] = Color(array[y, x] - 1)
                else:
                    self.grid[y][x] = None
        return self


# ----------------------------------------------------------------------------------
# Specialized grid generation strategies
# ----------------------------------------------------------------------------------

def generate_modulo_grid(size: int, params=None) -> LShapeGrid:
    """
    Generate a grid using modular arithmetic, which is known to work well
    for avoiding L-shapes.
    
    The pattern is of the form: color_idx = (a*x + b*y) % num_colors
    """
    grid = LShapeGrid(size)
    
    # Default parameters if none provided
    if params is None:
        a, b = 1, 2
    else:
        a, b = params
    
    # Apply the modular pattern
    for y in range(size):
        for x in range(size):
            color_idx = (a*x + b*y) % 3
            grid.set_color(x, y, list(Color)[color_idx])
    
    return grid


def generate_block_grid(size: int, block_size: int) -> LShapeGrid:
    """
    Generate a grid using block patterns, where each block is colored
    to avoid L-shapes within and between blocks.
    """
    grid = LShapeGrid(size)
    
    # Use different patterns for each block
    for block_y in range((size + block_size - 1) // block_size):
        for block_x in range((size + block_size - 1) // block_size):
            # Determine block pattern based on position
            pattern_type = (block_x + block_y) % 3
            
            # Fill the block with the appropriate pattern
            for y_offset in range(block_size):
                for x_offset in range(block_size):
                    x = block_x * block_size + x_offset
                    y = block_y * block_size + y_offset
                    
                    if x >= size or y >= size:
                        continue
                    
                    if pattern_type == 0:
                        # Diagonal pattern
                        color_idx = (x_offset + y_offset) % 3
                    elif pattern_type == 1:
                        # Reverse diagonal pattern
                        color_idx = (x_offset + block_size - 1 - y_offset) % 3
                    else:
                        # Spiral pattern
                        color_idx = ((x_offset + y_offset) * (x_offset - y_offset)) % 3
                    
                    grid.set_color(x, y, list(Color)[color_idx])
    
    return grid


def generate_recursive_grid(size: int) -> LShapeGrid:
    """
    Generate a grid using a recursive approach, where valid smaller grids
    are combined to form larger grids.
    """
    grid = LShapeGrid(size)
    
    # Base cases
    if size <= 3:
        # For 3x3, use a known valid configuration
        patterns = [
            [[0, 1, 2], [2, 0, 1], [1, 2, 0]],  # Latin square
            [[0, 1, 2], [1, 2, 0], [2, 0, 1]]   # Another Latin square
        ]
        pattern = random.choice(patterns)
        
        for y in range(min(size, 3)):
            for x in range(min(size, 3)):
                grid.set_color(x, y, list(Color)[pattern[y][x]])
    else:
        # Split into quadrants
        half = size // 2
        extra = size % 2
        
        # Generate each quadrant
        def fill_quadrant(grid, start_x, start_y, quadrant_size, pattern_shift):
            for y in range(quadrant_size):
                for x in range(quadrant_size):
                    # Use different pattern for each quadrant
                    color_idx = (x + y + pattern_shift) % 3
                    grid.set_color(start_x + x, start_y + y, list(Color)[color_idx])
        
        # Top-left quadrant
        fill_quadrant(grid, 0, 0, half + extra, 0)
        
        # Top-right quadrant
        fill_quadrant(grid, half + extra, 0, half, 1)
        
        # Bottom-left quadrant
        fill_quadrant(grid, 0, half + extra, half, 2)
        
        # Bottom-right quadrant
        fill_quadrant(grid, half + extra, half + extra, half, 0)
    
    return grid


def generate_quasi_periodicity_grid(size: int) -> LShapeGrid:
    """
    Generate a grid using a quasi-periodic pattern inspired by
    aperiodic tilings, which can avoid creating repeating patterns
    that might form L-shapes.
    """
    grid = LShapeGrid(size)
    
    # Golden ratio approximation
    phi = (1 + 5 ** 0.5) / 2
    
    for y in range(size):
        for x in range(size):
            # Generate quasi-periodic pattern based on golden ratio
            z = x * phi + y / phi
            color_idx = int(3 * (z - int(z)))
            grid.set_color(x, y, list(Color)[color_idx])
    
    return grid


def generate_latin_square_grid(size: int) -> LShapeGrid:
    """
    Generate a Latin square grid where each row and column contains
    each color exactly once (if size equals number of colors).
    This naturally avoids some L-shapes.
    """
    grid = LShapeGrid(size)
    
    # Initialize with a standard Latin square pattern
    for y in range(size):
        for x in range(size):
            color_idx = (x + y) % min(3, size)
            grid.set_color(x, y, list(Color)[color_idx])
    
    # For larger grids, extend the pattern
    if size > 3:
        for y in range(size):
            for x in range(size):
                if x >= 3 or y >= 3:
                    # Use a different pattern for extended part
                    if x >= 3 and y >= 3:
                        color_idx = (x + 2*y) % 3
                    elif x >= 3:
                        color_idx = (2*x + y) % 3
                    else:  # y >= 3
                        color_idx = ((x+1) * (y+1)) % 3
                    
                    grid.set_color(x, y, list(Color)[color_idx])
    
    return grid


# ----------------------------------------------------------------------------------
# Local search and optimization
# ----------------------------------------------------------------------------------

def local_search(grid: LShapeGrid, max_iterations: int = 1000) -> Tuple[LShapeGrid, bool]:
    """
    Perform local search to eliminate L-shapes by trying small changes to the grid.
    
    Returns:
        Tuple of (optimized grid, whether the grid is valid)
    """
    current_grid = grid.clone()
    best_grid = grid.clone()
    best_l_count = float('inf')
    
    # Count initial L-shapes
    has_l, l_points = current_grid.has_any_l_shape()
    if not has_l:
        return current_grid, True
    
    # Function to count L-shapes in the grid
    def count_l_shapes(g: LShapeGrid) -> int:
        count = 0
        for x in range(g.size):
            for y in range(g.size):
                has_l, _ = g.has_l_shape(x, y)
                if has_l:
                    count += 1
        return count
    
    current_l_count = count_l_shapes(current_grid)
    best_l_count = current_l_count
    
    # Perform local search
    for _ in range(max_iterations):
        # Clone the current grid
        temp_grid = current_grid.clone()
        
        # Choose a random cell
        x = random.randint(0, grid.size - 1)
        y = random.randint(0, grid.size - 1)
        
        # Try a different color
        current_color = temp_grid.get_color(x, y)
        new_color = random.choice([c for c in list(Color) if c != current_color])
        temp_grid.set_color(x, y, new_color)
        
        # Check if the change improves the grid
        temp_l_count = count_l_shapes(temp_grid)
        
        # Accept the change if it improves (less L-shapes)
        if temp_l_count <= current_l_count:
            current_grid = temp_grid
            current_l_count = temp_l_count
            
            # Update best if improved
            if current_l_count < best_l_count:
                best_grid = current_grid.clone()
                best_l_count = current_l_count
                
                # If no L-shapes, we're done
                if best_l_count == 0:
                    return best_grid, True
    
    return best_grid, best_l_count == 0


def genetic_algorithm(size: int, population_size: int = 20, generations: int = 50) -> Tuple[LShapeGrid, bool]:
    """
    Use a genetic algorithm approach to find a valid grid configuration.
    
    Returns:
        Tuple of (best grid, whether it's valid)
    """
    # Create initial population
    population = []
    
    # Generate initial population with different strategies
    for _ in range(population_size // 5):
        population.append(generate_modulo_grid(size, (1, 2)))
        population.append(generate_modulo_grid(size, (2, 1)))
        population.append(generate_block_grid(size, 2))
        population.append(generate_recursive_grid(size))
        population.append(generate_quasi_periodicity_grid(size))
    
    # Fill remaining slots if needed
    while len(population) < population_size:
        population.append(generate_latin_square_grid(size))
    
    # Function to evaluate a grid (lower is better)
    def evaluate_grid(grid: LShapeGrid) -> int:
        # Count L-shapes as a measure of fitness
        count = 0
        for x in range(grid.size):
            for y in range(grid.size):
                has_l, _ = grid.has_l_shape(x, y)
                if has_l:
                    count += 1
        return count
    
    best_grid = population[0].clone()
    best_score = evaluate_grid(best_grid)
    
    # For each generation
    for generation in range(generations):
        # Evaluate all grids
        scores = [evaluate_grid(grid) for grid in population]
        
        # Check if we have a valid solution
        if min(scores) == 0:
            best_index = scores.index(0)
            return population[best_index], True
        
        # Update best grid if improved
        if min(scores) < best_score:
            best_score = min(scores)
            best_index = scores.index(best_score)
            best_grid = population[best_index].clone()
        
        # Select parents (tournament selection)
        def select_parent():
            # Select 3 random individuals and choose the best
            indices = random.sample(range(population_size), 3)
            return population[min(indices, key=lambda i: scores[i])]
        
        # Create next generation
        next_population = []
        
        # Elitism: Keep the best individual
        best_index = scores.index(min(scores))
        next_population.append(population[best_index].clone())
        
        # Create offspring
        while len(next_population) < population_size:
            # Select parents
            parent1 = select_parent()
            parent2 = select_parent()
            
            # Crossover
            child = LShapeGrid(size)
            
            for y in range(size):
                for x in range(size):
                    # 50% chance to inherit from each parent
                    if random.random() < 0.5:
                        child.grid[y][x] = parent1.grid[y][x]
                    else:
                        child.grid[y][x] = parent2.grid[y][x]
            
            # Mutation (5% chance)
            for y in range(size):
                for x in range(size):
                    if random.random() < 0.05:
                        child.set_color(x, y, random.choice(list(Color)))
            
            next_population.append(child)
        
        # Replace with the new generation
        population = next_population
        
        # Optional: Print progress every 10 generations
        if generation % 10 == 0:
            print(f"  Generation {generation}: Best score = {best_score}")
    
    return best_grid, best_score == 0


# ----------------------------------------------------------------------------------
# Main solver function
# ----------------------------------------------------------------------------------

def solve_l_shape_ramsey(grid_sizes: List[int], attempts_per_size: int = 10, 
                         use_genetic: bool = True, max_local_search_iterations: int = 500,
                         visualize: bool = True) -> Dict[int, Tuple[LShapeGrid, bool]]:
    """
    Solve the L-shape Ramsey problem for multiple grid sizes
    
    Returns:
        Dictionary mapping grid size to (best grid, is valid)
    """
    # Create visualizations directory
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Results dictionary
    results = {}
    
    start_time = time.time()
    
    print(f"L-shape Ramsey Problem Solver")
    print(f"Grid sizes: {grid_sizes}")
    print(f"Attempts per size: {attempts_per_size}")
    print("=" * 50)
    
    for size in grid_sizes:
        print(f"\nSolving for grid size {size}x{size}")
        
        # Try different approaches
        best_grid = None
        is_valid = False
        
        # 1. First try the specialized strategies
        strategies = [
            ("Modulo (1,2)", lambda: generate_modulo_grid(size, (1, 2))),
            ("Modulo (2,1)", lambda: generate_modulo_grid(size, (2, 1))),
            ("Block Pattern", lambda: generate_block_grid(size, 2)),
            ("Recursive Pattern", lambda: generate_recursive_grid(size)),
            ("Quasi-periodicity", lambda: generate_quasi_periodicity_grid(size)),
            ("Latin Square", lambda: generate_latin_square_grid(size))
        ]
        
        for strategy_name, strategy_fn in strategies:
            print(f"  Trying {strategy_name} strategy...")
            grid = strategy_fn()
            has_l, l_points = grid.has_any_l_shape()
            
            if not has_l:
                print(f"  Success! {strategy_name} produced a valid grid with no L-shapes.")
                best_grid = grid
                is_valid = True
                break
        
        # 2. If no strategy worked directly, try local search
        if not is_valid:
            print(f"  No direct strategy worked. Trying local search optimization...")
            
            for i in range(attempts_per_size):
                # Choose a random strategy as starting point
                strategy_name, strategy_fn = random.choice(strategies)
                grid = strategy_fn()
                
                # Apply local search
                optimized_grid, is_valid_grid = local_search(grid, max_local_search_iterations)
                
                if is_valid_grid:
                    print(f"  Success! Local search found a valid grid on attempt {i+1}.")
                    best_grid = optimized_grid
                    is_valid = True
                    break
                
                # Keep track of the best result even if not valid
                if best_grid is None:
                    best_grid = optimized_grid
                else:
                    # Compare based on number of L-shapes
                    has_l1, _ = best_grid.has_any_l_shape()
                    has_l2, _ = optimized_grid.has_any_l_shape()
                    
                    if not has_l1 and has_l2:
                        pass  # Keep current best
                    elif has_l1 and not has_l2:
                        best_grid = optimized_grid  # New grid is better
        
        # 3. If still no valid solution and genetic algorithm enabled, try that
        if not is_valid and use_genetic and size <= 8:  # Genetic is expensive for large grids
            print(f"  Local search failed. Trying genetic algorithm...")
            genetic_grid, is_valid_genetic = genetic_algorithm(size)
            
            if is_valid_genetic:
                print(f"  Success! Genetic algorithm found a valid grid.")
                best_grid = genetic_grid
                is_valid = True
        
        # Store results
        results[size] = (best_grid, is_valid)
        
        # Save visualization
        if visualize and best_grid is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{viz_dir}/l_shape_grid_{size}x{size}_{timestamp}.png"
            best_grid.visualize(filename)
            print(f"  Visualization saved to {filename}")
        
        # Print result
        if is_valid:
            print(f"  Result for {size}x{size}: Valid solution found!")
        else:
            print(f"  Result for {size}x{size}: No valid solution found.")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Summary of results:")
    for size in grid_sizes:
        _, is_valid = results[size]
        result_str = "Valid solution" if is_valid else "No valid solution"
        print(f"Grid {size}x{size}: {result_str}")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f} seconds")
    
    return results


if __name__ == "__main__":
    # Solve for grid sizes 3-8
    grid_sizes = [3, 4, 5, 6, 7, 8]
    results = solve_l_shape_ramsey(grid_sizes, attempts_per_size=5, use_genetic=True) 