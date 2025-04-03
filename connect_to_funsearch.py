#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from datetime import datetime
import os
import random
from typing import List, Tuple, Optional

# Import the core FunSearch modules
from l_shape_ramsey import LShapeGrid, Color
# We'll adapt and reuse functions rather than import directly
import l_shape_funsearch as funsearch_module

# Import our deterministic solver for initial solutions
from fixed_solver import LShapeRamseySolver

class FunSearchConnector:
    """
    Connects the deterministic L-shape Ramsey solver with the FunSearch framework.
    This allows using our proven patterns as a starting point for further optimization.
    """
    
    def __init__(self):
        """Initialize the connector."""
        self.deterministic_solver = LShapeRamseySolver()
        self.visualizations_dir = "visualizations"
        os.makedirs(self.visualizations_dir, exist_ok=True)
    
    def deterministic_to_funsearch(self, np_grid: np.ndarray) -> LShapeGrid:
        """
        Convert a numpy array grid from our deterministic solver to a LShapeGrid
        used by the FunSearch framework.
        
        Args:
            np_grid: A numpy array with values 0, 1, 2
            
        Returns:
            A LShapeGrid object with the same coloring
        """
        size = np_grid.shape[0]
        grid = LShapeGrid(size)
        
        for i in range(size):
            for j in range(size):
                # Map 0, 1, 2 to Color.RED, Color.GREEN, Color.BLUE
                color_value = np_grid[i, j]
                color = list(Color)[color_value]
                grid.set_color(j, i, color)  # Note: LShapeGrid uses (x,y) order
        
        return grid
    
    def funsearch_to_deterministic(self, funsearch_grid: LShapeGrid) -> np.ndarray:
        """
        Convert a LShapeGrid from FunSearch to a numpy array used by our deterministic solver.
        
        Args:
            funsearch_grid: A LShapeGrid object
            
        Returns:
            A numpy array with values 0, 1, 2
        """
        size = funsearch_grid.size
        np_grid = np.zeros((size, size), dtype=int)
        
        for i in range(size):
            for j in range(size):
                color = funsearch_grid.get_color(j, i)  # Note: LShapeGrid uses (x,y) order
                if color is not None:
                    np_grid[i, j] = color.value
        
        return np_grid
    
    def seeded_funsearch(self, size: int, seed_grid: LShapeGrid, max_iterations: int = 1000, time_limit: int = 300) -> Tuple[LShapeGrid, float]:
        """
        Run FunSearch with a specific seed grid.
        This is a modified version of the funsearch function from l_shape_funsearch.py.
        
        Args:
            size: Size of the grid
            seed_grid: Initial grid to start from
            max_iterations: Maximum number of iterations
            time_limit: Time limit in seconds
            
        Returns:
            The best grid found and its score
        """
        best_grid = None
        best_score = 0
        
        # Start with our seed grid and create variations
        population_size = 100
        population = [seed_grid]
        
        # Add mutated versions of our solution to the population
        for _ in range(population_size - 1):
            # Copy the grid
            mutated = LShapeGrid(size)
            for i in range(size):
                for j in range(size):
                    mutated.set_color(j, i, seed_grid.get_color(j, i))
            
            # Mutate a few cells
            num_changes = random.randint(1, max(2, size // 2))
            for _ in range(num_changes):
                x = random.randrange(size)
                y = random.randrange(size)
                current_color = mutated.get_color(x, y)
                available_colors = [c for c in Color if c != current_color]
                mutated.set_color(x, y, random.choice(available_colors))
            
            population.append(mutated)
        
        # The rest of the function follows the original funsearch implementation
        start_time = time.time()
        iteration = 0
        
        while iteration < max_iterations and (time.time() - start_time) < time_limit:
            # Evaluate current population
            scores = [funsearch_module.evaluate_grid(grid) for grid in population]
            
            # Update best solution
            max_score = max(scores)
            if max_score > best_score:
                best_score = max_score
                best_grid = population[scores.index(max_score)]
                print(f"Iteration {iteration}: Found better solution with score {best_score}")
                has_l, points = best_grid.has_any_l_shape()
                if not has_l:
                    print("Found valid solution!")
                    best_grid.visualize()
                    break
            
            # Create new population
            new_population = []
            
            # Keep best solutions (elitism)
            elite_size = max(2, population_size // 10)
            elite_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:elite_size]
            new_population.extend([population[i] for i in elite_indices])
            
            # Fill rest with combination of mutation and crossover
            while len(new_population) < population_size:
                if random.random() < 0.7:  # 70% chance of crossover
                    # Tournament selection for parents
                    tournament_size = 5
                    parent1_idx = max(random.sample(range(len(population)), tournament_size), 
                                    key=lambda i: scores[i])
                    parent2_idx = max(random.sample(range(len(population)), tournament_size), 
                                    key=lambda i: scores[i])
                    
                    child = funsearch_module.crossover(population[parent1_idx], population[parent2_idx])
                    # Apply mutation to crossover result
                    if random.random() < 0.3:  # 30% chance of mutation after crossover
                        child = funsearch_module.mutate_grid(child)
                else:  # 30% chance of mutation only
                    # Tournament selection for parent
                    tournament_size = 3
                    parent_idx = max(random.sample(range(len(population)), tournament_size), 
                                   key=lambda i: scores[i])
                    child = funsearch_module.mutate_grid(population[parent_idx])
                
                new_population.append(child)
            
            population = new_population
            iteration += 1
            
            # Print progress every 100 iterations
            if iteration % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"Iteration {iteration}, Best score: {best_score}, Time elapsed: {elapsed_time:.1f}s")
        
        return best_grid, best_score
    
    def run_funsearch_with_deterministic_seed(self, grid_size: int, iterations: int = 1000, time_limit: int = 300):
        """
        Run FunSearch using our deterministic solution as a starting point.
        
        Args:
            grid_size: Size of the grid
            iterations: Maximum number of iterations
            time_limit: Time limit in seconds
        """
        print(f"Generating deterministic solution for {grid_size}×{grid_size} grid...")
        
        # First, get a deterministic solution
        deterministic_solution = self.deterministic_solver.generate_solution(grid_size)
        
        # Verify the deterministic solution
        is_valid = self.deterministic_solver.verify_solution(deterministic_solution)
        print(f"Deterministic solution is valid: {is_valid}")
        
        if is_valid:
            # Convert to FunSearch format
            funsearch_grid = self.deterministic_to_funsearch(deterministic_solution)
            
            # Verify conversion
            has_l, points = funsearch_grid.has_any_l_shape()
            print(f"Converted solution has L-shape: {has_l}")
            
            if not has_l:
                # Now run FunSearch with this grid as a starting point
                print(f"Running FunSearch with deterministic solution as seed...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save the deterministic solution visualization
                self.deterministic_solver.visualize_solution(
                    deterministic_solution, grid_size, f"deterministic_seed_{timestamp}"
                )
                
                # Run our modified FunSearch with the seed grid
                best_grid, score = self.seeded_funsearch(
                    grid_size, 
                    funsearch_grid,
                    max_iterations=iterations, 
                    time_limit=time_limit
                )
                
                if best_grid is not None:
                    # Save the best solution found by FunSearch
                    has_l, points = best_grid.has_any_l_shape()
                    print(f"FunSearch found valid solution: {not has_l}")
                    print(f"Score: {score}")
                    
                    if not has_l:
                        # Convert back to our format
                        np_grid = self.funsearch_to_deterministic(best_grid)
                        
                        # Save visualization
                        self.deterministic_solver.visualize_solution(
                            np_grid, grid_size, f"funsearch_optimized_{timestamp}"
                        )
                        
                        print("FunSearch successfully improved the solution!")
                        print(np_grid)
                
            else:
                print("Error in conversion, cannot use as seed for FunSearch.")
        else:
            print("Deterministic solution is not valid, cannot use as seed for FunSearch.")

def main():
    parser = argparse.ArgumentParser(description='Connect deterministic solver with FunSearch')
    parser.add_argument('--grid-sizes', type=int, nargs='+', default=[3, 4, 5],
                        help='Grid sizes to solve (default: 3 4 5)')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Maximum iterations for FunSearch (default: 1000)')
    parser.add_argument('--time-limit', type=int, default=300,
                        help='Time limit in seconds for FunSearch (default: 300)')
    args = parser.parse_args()
    
    connector = FunSearchConnector()
    
    for size in args.grid_sizes:
        print(f"\n{'='*50}\nSolving {size}×{size} grid\n{'='*50}")
        connector.run_funsearch_with_deterministic_seed(
            size, 
            iterations=args.iterations,
            time_limit=args.time_limit
        )

if __name__ == "__main__":
    main() 