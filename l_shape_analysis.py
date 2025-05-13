from l_shape_ramsey import LShapeGrid, Color
import random
import numpy as np
from typing import List, Set, Tuple
import time
from collections import Counter
import hashlib

class GridAnalyzer:
    def __init__(self, size: int):
        self.size = size
        self.valid_solutions = set()  # Store hashes of valid solutions
        self.best_partial_solutions = []  # Store promising partial solutions
    
    def grid_to_hash(self, grid: LShapeGrid) -> str:
        """Convert grid to a unique hash string"""
        colors = [[grid.get_color(x, y).value for x in range(grid.size)] 
                 for y in range(grid.size)]
        return hashlib.sha256(str(colors).encode()).hexdigest()
    
    def analyze_solution(self, grid: LShapeGrid) -> dict:
        """Analyze properties of a valid solution"""
        stats = {
            'color_counts': Counter(),
            'row_diversity': [],
            'col_diversity': [],
            'adjacent_same_color': 0
        }
        
        # Count colors and analyze patterns
        for y in range(grid.size):
            row_colors = set()
            col_colors = set()
            for x in range(grid.size):
                color = grid.get_color(x, y)
                stats['color_counts'][color] += 1
                row_colors.add(color)
                
                # Check adjacent cells
                if x > 0 and grid.get_color(x-1, y) == color:
                    stats['adjacent_same_color'] += 1
                if y > 0 and grid.get_color(x, y-1) == color:
                    stats['adjacent_same_color'] += 1
            
            stats['row_diversity'].append(len(row_colors))
        
        # Column diversity
        for x in range(grid.size):
            col_colors = set(grid.get_color(x, y) for y in range(grid.size))
            stats['col_diversity'].append(len(col_colors))
        
        return stats
    
    def estimate_bounds(self) -> Tuple[float, float]:
        """
        Estimate lower and upper bounds for number of valid solutions
        Uses probabilistic methods and pattern analysis
        """
        total_cells = self.size * self.size
        total_configurations = 3 ** total_cells
        
        # Estimate probability of L-shape occurrence
        l_shape_prob = 0
        for x in range(self.size):
            for y in range(self.size):
                # Count possible L-shapes from this point
                possible_ls = 0
                for d in range(1, self.size):
                    # Right and Up
                    if x + d < self.size and y + d < self.size:
                        possible_ls += 1
                    # Right and Down
                    if x + d < self.size and y - d >= 0:
                        possible_ls += 1
                    # Left and Up
                    if x - d >= 0 and y + d < self.size:
                        possible_ls += 1
                    # Left and Down
                    if x - d >= 0 and y - d >= 0:
                        possible_ls += 1
                
                # Probability of monochromatic L-shape at this point
                if possible_ls > 0:
                    l_shape_prob += possible_ls * (1/3)**3  # Probability of same color for 3 points
        
        # Rough upper bound: configurations without obvious L-shapes
        upper_bound = total_configurations * (1 - l_shape_prob)
        
        # Lower bound: based on known construction methods
        # Using block construction method
        block_size = 4  # Minimum size that usually allows valid configurations
        num_blocks = (self.size // block_size) ** 2
        lower_bound = 3 ** num_blocks  # Each block can be one of 3 patterns
        
        return lower_bound, upper_bound
    
    def find_solutions(self, time_limit: int = 3600, max_solutions: int = 100) -> List[LShapeGrid]:
        """
        Try to find multiple valid solutions using different search strategies
        """
        solutions = []
        start_time = time.time()
        
        # Initialize different starting configurations
        strategies = [
            self._generate_block_based,
            self._generate_diagonal_based,
            self._generate_random
        ]
        
        while (time.time() - start_time < time_limit and 
               len(solutions) < max_solutions):
            
            # Try different generation strategies
            for strategy in strategies:
                grid = strategy()
                if not grid.has_any_l_shape()[0]:
                    grid_hash = self.grid_to_hash(grid)
                    if grid_hash not in self.valid_solutions:
                        self.valid_solutions.add(grid_hash)
                        solutions.append(grid)
                        print(f"Found solution {len(solutions)}")
                        grid.visualize()
                
                if len(solutions) >= max_solutions:
                    break
            
            if len(solutions) == 0:
                print("No solutions found yet, continuing search...")
        
        return solutions
    
    def _generate_block_based(self) -> LShapeGrid:
        """Generate a grid using block-based construction"""
        grid = LShapeGrid(self.size)
        block_size = 4
        
        for by in range(0, self.size, block_size):
            for bx in range(0, self.size, block_size):
                # Fill block with a valid pattern
                colors = list(Color)
                random.shuffle(colors)
                for y in range(by, min(by + block_size, self.size)):
                    for x in range(bx, min(bx + block_size, self.size)):
                        color_idx = (x - bx + y - by) % 3
                        grid.set_color(x, y, colors[color_idx])
        
        return grid
    
    def _generate_diagonal_based(self) -> LShapeGrid:
        """Generate a grid using diagonal stripes"""
        grid = LShapeGrid(self.size)
        
        for y in range(self.size):
            for x in range(self.size):
                diagonal = (x + y) % 3
                grid.set_color(x, y, Color(diagonal))
        
        return grid
    
    def _generate_random(self) -> LShapeGrid:
        """Generate a random grid"""
        grid = LShapeGrid(self.size)
        colors = list(Color)
        
        for y in range(self.size):
            for x in range(self.size):
                grid.set_color(x, y, random.choice(colors))
        
        return grid

def main():
    size = 6
    analyzer = GridAnalyzer(size)
    
    # Estimate bounds
    lower_bound, upper_bound = analyzer.estimate_bounds()
    print(f"\nEstimated bounds for {size}x{size} grid:")
    print(f"Lower bound: {lower_bound:.2e} configurations")
    print(f"Upper bound: {upper_bound:.2e} configurations")
    
    # Try to find some solutions
    print(f"\nSearching for valid configurations...")
    solutions = analyzer.find_solutions(time_limit=600, max_solutions=5)  # 10 minutes, 5 solutions
    
    print(f"\nFound {len(solutions)} valid configurations")
    print(f"Total unique solutions found: {len(analyzer.valid_solutions)}")
    
    if solutions:
        print("\nAnalyzing solutions:")
        for i, solution in enumerate(solutions):
            print(f"\nSolution {i+1} analysis:")
            stats = analyzer.analyze_solution(solution)
            print("Color distribution:", dict(stats['color_counts']))
            print("Average row diversity:", sum(stats['row_diversity']) / len(stats['row_diversity']))
            print("Average col diversity:", sum(stats['col_diversity']) / len(stats['col_diversity']))
            print("Adjacent same color count:", stats['adjacent_same_color'])

if __name__ == "__main__":
    main() 