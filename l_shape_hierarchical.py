from l_shape_ramsey import LShapeGrid, Color
import random
from typing import List, Dict, Set, Tuple
import time
import hashlib

class HierarchicalGridBuilder:
    def __init__(self):
        self.known_solutions: Dict[int, List[LShapeGrid]] = {}
        self.solution_hashes: Dict[int, Set[str]] = {}
    
    def grid_to_hash(self, grid: LShapeGrid) -> str:
        """Convert grid to a unique hash string"""
        colors = [[grid.get_color(x, y).value for x in range(grid.size)] 
                 for y in range(grid.size)]
        return hashlib.sha256(str(colors).encode()).hexdigest()
    
    def copy_subgrid(self, source: LShapeGrid, target: LShapeGrid, 
                     src_x: int, src_y: int, tgt_x: int, tgt_y: int, size: int):
        """Copy a subgrid from source to target"""
        for y in range(size):
            for x in range(size):
                color = source.get_color(src_x + x, src_y + y)
                target.set_color(tgt_x + x, tgt_y + y, color)
    
    def find_base_solutions(self, size: int, time_limit: int = 300) -> List[LShapeGrid]:
        """Find valid solutions for a base size grid"""
        if size in self.known_solutions:
            return self.known_solutions[size]
        
        solutions = []
        start_time = time.time()
        self.solution_hashes[size] = set()
        
        while time.time() - start_time < time_limit and len(solutions) < 10:
            grid = self._generate_candidate(size)
            if not grid.has_any_l_shape()[0]:
                grid_hash = self.grid_to_hash(grid)
                if grid_hash not in self.solution_hashes[size]:
                    self.solution_hashes[size].add(grid_hash)
                    solutions.append(grid)
                    # Save visualization instead of displaying
                    filename = self.save_grid_visualization(grid, size, "base", len(solutions))
                    print(f"Found valid {size}x{size} solution {len(solutions)}, saved as {filename}")
        
        self.known_solutions[size] = solutions
        return solutions
    
    def _generate_candidate(self, size: int) -> LShapeGrid:
        """Generate a candidate grid using various strategies"""
        grid = LShapeGrid(size)
        
        # Try different generation strategies
        strategies = [
            self._generate_diagonal,
            self._generate_cyclic,
            self._generate_random
        ]
        
        strategy = random.choice(strategies)
        return strategy(size)
    
    def _generate_diagonal(self, size: int) -> LShapeGrid:
        """Generate grid with diagonal pattern"""
        grid = LShapeGrid(size)
        colors = list(Color)
        random.shuffle(colors)
        
        for y in range(size):
            for x in range(size):
                color_idx = (x + y) % 3
                grid.set_color(x, y, colors[color_idx])
        return grid
    
    def _generate_cyclic(self, size: int) -> LShapeGrid:
        """Generate grid with cyclic pattern"""
        grid = LShapeGrid(size)
        colors = list(Color)
        random.shuffle(colors)
        
        for y in range(size):
            for x in range(size):
                color_idx = (x + 2*y) % 3
                grid.set_color(x, y, colors[color_idx])
        return grid
    
    def _generate_random(self, size: int) -> LShapeGrid:
        """Generate random grid"""
        grid = LShapeGrid(size)
        colors = list(Color)
        
        for y in range(size):
            for x in range(size):
                grid.set_color(x, y, random.choice(colors))
        return grid
    
    def build_larger_grid(self, target_size: int, base_size: int = 5) -> LShapeGrid:
        """
        Build a larger grid using known valid smaller grids.
        Uses base_size x base_size valid grids as building blocks.
        """
        # First ensure we have solutions for the base size
        base_solutions = self.find_base_solutions(base_size)
        if not base_solutions:
            raise ValueError(f"No valid {base_size}x{base_size} solutions found")
        
        # Create the larger grid
        grid = LShapeGrid(target_size)
        
        # Fill the grid block by block
        for by in range(0, target_size, base_size):
            for bx in range(0, target_size, base_size):
                # Select a random valid base solution
                base = random.choice(base_solutions)
                
                # Calculate the actual size for this block (might be smaller at edges)
                block_width = min(base_size, target_size - bx)
                block_height = min(base_size, target_size - by)
                
                # Try different rotations and transformations of the base solution
                attempts = 0
                while attempts < 10:
                    # Copy the block
                    self.copy_subgrid(base, grid, 0, 0, bx, by, min(block_width, block_height))
                    
                    # Check if this creates any L-shapes
                    has_l = False
                    for y in range(max(0, by-1), min(target_size, by+block_height+1)):
                        for x in range(max(0, bx-1), min(target_size, bx+block_width+1)):
                            if grid.has_l_shape(x, y)[0]:
                                has_l = True
                                break
                        if has_l:
                            break
                    
                    if not has_l:
                        break
                    
                    # If we created L-shapes, try a different base solution
                    base = random.choice(base_solutions)
                    attempts += 1
                
                if attempts == 10:
                    print(f"Warning: Could not find valid arrangement for block at ({bx},{by})")
        
        return grid

    def save_grid_visualization(self, grid, size: int, strategy: str, index: int):
        """Save a grid visualization to the visualizations directory"""
        filename = f"visualizations/grid_{size}x{size}_{strategy}_{index}.png"
        grid.visualize(filename=filename)
        return filename

    def find_solutions(self, size: int, max_solutions: int = 10, time_limit: int = 300):
        """Find solutions for a given grid size using various strategies"""
        solutions = []
        start_time = time.time()
        
        for strategy in ['diagonal', 'cyclic', 'random']:
            strategy_solutions = 0
            while len(solutions) < max_solutions and time.time() - start_time < time_limit:
                # Use the correct generation method based on strategy
                if strategy == 'diagonal':
                    grid = self._generate_diagonal(size)
                elif strategy == 'cyclic':
                    grid = self._generate_cyclic(size)
                else:  # random
                    grid = self._generate_random(size)
                
                if not grid.has_any_l_shape()[0]:
                    grid_hash = self.grid_to_hash(grid)
                    if grid_hash not in self.solution_hashes.get(size, set()):
                        if size not in self.solution_hashes:
                            self.solution_hashes[size] = set()
                        self.solution_hashes[size].add(grid_hash)
                        solutions.append(grid)
                        strategy_solutions += 1
                        # Save visualization
                        filename = self.save_grid_visualization(grid, size, strategy, strategy_solutions)
                        print(f"Found valid solution {len(solutions)}, saved as {filename}")
        
        return solutions

def main():
    builder = HierarchicalGridBuilder()
    
    # First find solutions for base sizes
    print("Finding solutions for base sizes...")
    for size in range(4, 7):
        print(f"\nSearching for {size}x{size} solutions:")
        solutions = builder.find_base_solutions(size)
        print(f"Found {len(solutions)} valid {size}x{size} configurations")
    
    # Try to build larger grids
    target_sizes = [10, 15, 20]
    for size in target_sizes:
        print(f"\nAttempting to build {size}x{size} grid...")
        try:
            large_grid = builder.build_larger_grid(size)
            has_l, points = large_grid.has_any_l_shape()
            if has_l:
                print(f"Warning: {size}x{size} grid contains L-shapes at {points}")
            else:
                print(f"Successfully built valid {size}x{size} grid!")
                large_grid.visualize()
        except Exception as e:
            print(f"Error building {size}x{size} grid: {e}")

if __name__ == "__main__":
    main() 