from l_shape_ramsey import LShapeGrid, Color
import random
from typing import List, Tuple
import time

def generate_random_grid(size: int) -> LShapeGrid:
    """Generate a random grid coloring"""
    grid = LShapeGrid(size)
    colors = list(Color)
    for x in range(size):
        for y in range(size):
            grid.set_color(x, y, random.choice(colors))
    return grid

def evaluate_grid(grid: LShapeGrid) -> float:
    """
    Evaluate how good a grid configuration is.
    Returns a score where:
    - Higher is better
    - 0 means invalid (has L-shape)
    - Positive values indicate valid configurations
    """
    has_l, _ = grid.has_any_l_shape()
    if has_l:
        return 0
    
    # Count color diversity and patterns
    score = 0
    
    # Reward color diversity in rows and columns
    for i in range(grid.size):
        row_colors = set()
        col_colors = set()
        for j in range(grid.size):
            row_colors.add(grid.get_color(j, i))
            col_colors.add(grid.get_color(i, j))
        
        # More diverse rows/columns get higher scores
        score += len(row_colors) + len(col_colors)
    
    # Penalize adjacent same colors (to discourage potential L-shapes)
    for x in range(grid.size):
        for y in range(grid.size):
            color = grid.get_color(x, y)
            # Check right neighbor
            if x < grid.size - 1 and grid.get_color(x + 1, y) == color:
                score -= 0.5
            # Check bottom neighbor
            if y < grid.size - 1 and grid.get_color(x, y + 1) == color:
                score -= 0.5
    
    return score

def mutate_grid(grid: LShapeGrid) -> LShapeGrid:
    """Create a slightly modified version of the grid"""
    new_grid = LShapeGrid(grid.size)
    
    # Copy existing grid
    for x in range(grid.size):
        for y in range(grid.size):
            new_grid.set_color(x, y, grid.get_color(x, y))
    
    # Randomly change a few cells
    num_changes = random.randint(1, max(2, grid.size // 2))
    for _ in range(num_changes):
        x = random.randrange(grid.size)
        y = random.randrange(grid.size)
        # Avoid using the same color that's currently there
        current_color = new_grid.get_color(x, y)
        available_colors = [c for c in Color if c != current_color]
        new_grid.set_color(x, y, random.choice(available_colors))
    
    return new_grid

def crossover(grid1: LShapeGrid, grid2: LShapeGrid) -> LShapeGrid:
    """Create a new grid by combining two parent grids"""
    if grid1.size != grid2.size:
        raise ValueError("Grids must be the same size")
    
    child = LShapeGrid(grid1.size)
    
    # Randomly choose sections from each parent
    for x in range(grid1.size):
        for y in range(grid1.size):
            if random.random() < 0.5:
                child.set_color(x, y, grid1.get_color(x, y))
            else:
                child.set_color(x, y, grid2.get_color(x, y))
    
    return child

def funsearch(size: int, max_iterations: int = 10000, time_limit: int = 300) -> Tuple[LShapeGrid, float]:
    """
    Run FunSearch to find a valid grid configuration.
    Returns the best grid found and its score.
    
    Args:
        size: Size of the grid
        max_iterations: Maximum number of iterations
        time_limit: Time limit in seconds
    """
    best_grid = None
    best_score = 0
    
    # Start with random grids
    population_size = 100
    population = [generate_random_grid(size) for _ in range(population_size)]
    
    start_time = time.time()
    iteration = 0
    
    while iteration < max_iterations and (time.time() - start_time) < time_limit:
        # Evaluate current population
        scores = [evaluate_grid(grid) for grid in population]
        
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
                
                child = crossover(population[parent1_idx], population[parent2_idx])
                # Apply mutation to crossover result
                if random.random() < 0.3:  # 30% chance of mutation after crossover
                    child = mutate_grid(child)
            else:  # 30% chance of mutation only
                # Tournament selection for parent
                tournament_size = 3
                parent_idx = max(random.sample(range(len(population)), tournament_size), 
                               key=lambda i: scores[i])
                child = mutate_grid(population[parent_idx])
            
            new_population.append(child)
        
        population = new_population
        iteration += 1
        
        # Print progress every 100 iterations
        if iteration % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Iteration {iteration}, Best score: {best_score}, Time elapsed: {elapsed_time:.1f}s")
    
    return best_grid, best_score

def main():
    # Try to find a valid 5x5 grid
    print("Searching for valid 5x5 grid...")
    best_grid, score = funsearch(5, max_iterations=10000, time_limit=300)
    
    if best_grid is not None:
        print("\nBest solution found:")
        print(best_grid)
        print(f"Score: {score}")
        has_l, points = best_grid.has_any_l_shape()
        print(f"Has L-shape: {has_l}")
        if has_l:
            print(f"L-shape points: {points}")
        best_grid.visualize()
    else:
        print("No valid solution found")

if __name__ == "__main__":
    main() 