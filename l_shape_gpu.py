import torch
from enum import Enum
from typing import List, Tuple, Optional, Dict, Set
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from l_shape_ramsey import Color
import time
import random
import hashlib
import psutil
import os

def get_device():
    """Get the appropriate device for PyTorch operations"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_gpu_memory_usage():
    """Get GPU memory usage in GB"""
    if torch.backends.mps.is_available():
        # For MPS, we can only get total memory through torch.mps.current_allocated_memory()
        return torch.mps.current_allocated_memory() / (1024**3)
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0

class LShapeGridGPU:
    def __init__(self, size: int):
        self.size = size
        self.device = get_device()
        self.start_time = time.time()
        
        # Initialize grid as a tensor on GPU
        self.grid = torch.full((size, size), -1, dtype=torch.int32, device=self.device)
        
        # Pre-compute L-shape patterns for faster detection
        self._init_l_patterns()
    
    def _init_l_patterns(self):
        """Initialize L-shape patterns for GPU-accelerated detection"""
        patterns = []
        # For a 4x4 grid, we only need patterns up to size 3
        max_size = min(self.size - 1, 3)  # Limit pattern size for smaller grids
        
        for d in range(1, max_size + 1):
            # Right and Up
            patterns.append([(0, 0), (d, 0), (d, d)])
            # Right and Down
            patterns.append([(0, 0), (d, 0), (d, -d)])
            # Left and Up
            patterns.append([(0, 0), (-d, 0), (-d, d)])
            # Left and Down
            patterns.append([(0, 0), (-d, 0), (-d, -d)])
        
        # Convert patterns to tensor for GPU computation
        self.patterns = torch.tensor(patterns, device=self.device)
        print(f"Initialized {len(patterns)} L-shape patterns for {self.size}x{self.size} grid")
    
    def set_color(self, x: int, y: int, color: Color) -> None:
        """Set the color at position (x,y)"""
        if 0 <= x < self.size and 0 <= y < self.size:
            self.grid[y, x] = color.value
    
    def get_color(self, x: int, y: int) -> Optional[Color]:
        """Get the color at position (x,y)"""
        if 0 <= x < self.size and 0 <= y < self.size:
            value = self.grid[y, x].item()
            return Color(value) if value >= 0 else None
        return None
    
    def has_l_shape(self, x: int, y: int) -> Tuple[bool, List[Tuple[int, int]]]:
        """GPU-accelerated L-shape detection"""
        color = self.get_color(x, y)
        if color is None:
            return False, []
        
        # Create a window around the point using PyTorch operations
        window_size = min(self.size, 5)  # Limit window size for efficiency
        window = torch.zeros((window_size, window_size), dtype=torch.int32, device=self.device)
        
        # Calculate window boundaries
        start_y = max(0, y - window_size//2)
        end_y = min(self.size, y + window_size//2)
        start_x = max(0, x - window_size//2)
        end_x = min(self.size, x + window_size//2)
        
        window_y_start = window_size//2 - (y - start_y)
        window_y_end = window_y_start + (end_y - start_y)
        window_x_start = window_size//2 - (x - start_x)
        window_x_end = window_x_start + (end_x - start_x)
        
        # Copy grid portion to window using PyTorch operations
        window[window_y_start:window_y_end, window_x_start:window_x_end] = \
            self.grid[start_y:end_y, start_x:end_x]
        
        # Check all patterns in parallel
        center_y, center_x = window_size//2, window_size//2
        center = torch.tensor([center_y, center_x], device=self.device)
        
        for pattern in self.patterns:
            points = pattern + center
            # Check if points are within window bounds
            if torch.any(points < 0) or torch.any(points >= window_size):
                continue
                
            colors = window[points[:, 0], points[:, 1]]
            if torch.all(colors == color.value):
                # Convert back to original coordinates using PyTorch operations
                original_points = points - center + torch.tensor([y, x], device=self.device)
                return True, [(p[1].item(), p[0].item()) for p in original_points]
        
        return False, []
    
    def has_any_l_shape(self) -> Tuple[bool, List[Tuple[int, int]]]:
        """Check if the grid contains any monochromatic L-shape"""
        # Process in batches for better GPU utilization
        batch_size = 1024  # Adjust based on GPU memory
        total_positions = self.size * self.size
        
        for i in range(0, total_positions, batch_size):
            batch_end = min(i + batch_size, total_positions)
            for j in range(i, batch_end):
                y = j // self.size
                x = j % self.size
                if self.get_color(x, y) is not None:
                    has_l, points = self.has_l_shape(x, y)
                    if has_l:
                        return True, points
        
        return False, []
    
    def visualize(self, highlight_l_shape: bool = True, filename: str = None):
        """Visualize the grid (CPU operation)"""
        # Convert grid to CPU for visualization using PyTorch operations
        grid_cpu = self.grid.cpu()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw grid using PyTorch tensor operations
        for y in range(self.size):
            for x in range(self.size):
                color_value = grid_cpu[y, x].item()
                if color_value >= 0:
                    color = Color(color_value)
                    rect = patches.Rectangle(
                        (x, self.size - 1 - y), 1, 1,
                        facecolor=color.name.lower(),
                        alpha=0.7,
                        edgecolor='black',
                        linewidth=1
                    )
                    ax.add_patch(rect)
        
        # Draw grid lines
        for i in range(self.size + 1):
            ax.axhline(y=i, color='black', linewidth=1)
            ax.axvline(x=i, color='black', linewidth=1)
        
        # Add coordinates
        for y in range(self.size):
            for x in range(self.size):
                bbox = dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)
                ax.text(x + 0.5, self.size - 0.5 - y, f'({x},{y})',
                       ha='center', va='center', fontsize=9,
                       bbox=bbox)
        
        if highlight_l_shape:
            has_l, points = self.has_any_l_shape()
            if has_l:
                plot_points = [(x + 0.5, self.size - 0.5 - y) for x, y in points]
                l_shape = plt.Polygon(plot_points, fill=False, color='black', linewidth=2)
                ax.add_patch(l_shape)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        plt.title("L-shape Ramsey Grid")
        ax.set_aspect('equal')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

class HierarchicalGridBuilderGPU:
    def __init__(self):
        self.known_solutions: Dict[int, List[LShapeGridGPU]] = {}
        self.solution_hashes: Dict[int, Set[str]] = {}
        self.device = get_device()
        self.total_attempts = 0
        self.successful_attempts = 0
        self.grid_cache = {}  # Cache for grid instances
        print(f"Initialized HierarchicalGridBuilderGPU on {self.device}")
    
    def _get_grid(self, size: int) -> LShapeGridGPU:
        """Get or create a grid instance for the given size"""
        if size not in self.grid_cache:
            self.grid_cache[size] = LShapeGridGPU(size)
        return self.grid_cache[size]
    
    def find_base_solutions(self, size: int, time_limit: int = 300) -> List[LShapeGridGPU]:
        """Find valid solutions for a base size grid using GPU acceleration"""
        if size in self.known_solutions:
            return self.known_solutions[size]
        
        solutions = []
        start_time = time.time()
        self.solution_hashes[size] = set()
        attempts = 0
        last_success_time = start_time
        no_progress_threshold = 30  # Reduced threshold for faster feedback
        
        print(f"\nSearching for {size}x{size} solutions:")
        print(f"Time limit: {time_limit} seconds")
        print(f"Target solutions: 10")
        
        # Get a reusable grid instance
        grid = self._get_grid(size)
        
        # Try some known valid patterns first
        known_patterns = [
            self._generate_diagonal,
            self._generate_cyclic,
            self._generate_checkerboard
        ]
        
        for pattern in known_patterns:
            pattern(grid)
            if not grid.has_any_l_shape()[0]:
                grid_hash = self.grid_to_hash(grid)
                if grid_hash not in self.solution_hashes[size]:
                    self.solution_hashes[size].add(grid_hash)
                    solution_grid = LShapeGridGPU(size)
                    solution_grid.grid = grid.grid.clone()
                    solutions.append(solution_grid)
                    self.successful_attempts += 1
                    last_success_time = time.time()
                    print(f"\nFound valid {size}x{size} solution using known pattern!")
                    filename = f"visualizations/grid_{size}x{size}_base_{len(solutions)}.png"
                    solution_grid.visualize(filename=filename)
                    print(f"Saved as: {filename}")
        
        while time.time() - start_time < time_limit and len(solutions) < 10:
            attempts += 1
            
            if time.time() - last_success_time > no_progress_threshold:
                print(f"\nNo progress for {no_progress_threshold} seconds. Stopping search.")
                break
            
            if attempts % 100 == 0:
                elapsed = time.time() - start_time
                gpu_mem = get_gpu_memory_usage()
                success_rate = (self.successful_attempts / max(1, self.total_attempts)) * 100
                print(f"\rProgress: {len(solutions)}/10 solutions | "
                      f"Attempts: {attempts} | "
                      f"Success rate: {success_rate:.1f}% | "
                      f"Time: {elapsed:.1f}s | "
                      f"GPU Memory: {gpu_mem:.1f}GB | "
                      f"Time since last success: {time.time() - last_success_time:.1f}s", end="")
            
            self._generate_candidate(grid)
            self.total_attempts += 1
            
            if not grid.has_any_l_shape()[0]:
                grid_hash = self.grid_to_hash(grid)
                if grid_hash not in self.solution_hashes[size]:
                    self.solution_hashes[size].add(grid_hash)
                    solution_grid = LShapeGridGPU(size)
                    solution_grid.grid = grid.grid.clone()
                    solutions.append(solution_grid)
                    self.successful_attempts += 1
                    last_success_time = time.time()
                    
                    filename = f"visualizations/grid_{size}x{size}_base_{len(solutions)}.png"
                    solution_grid.visualize(filename=filename)
                    print(f"\nFound valid {size}x{size} solution {len(solutions)}!")
                    print(f"Time taken: {time.time() - start_time:.1f}s")
                    print(f"Total attempts: {attempts}")
                    print(f"Saved as: {filename}")
        
        elapsed = time.time() - start_time
        print(f"\nCompleted {size}x{size} search:")
        print(f"Found {len(solutions)} solutions")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Average time per solution: {elapsed/len(solutions):.1f}s" if solutions else "No solutions found")
        print(f"Success rate: {(self.successful_attempts/self.total_attempts)*100:.1f}%")
        
        self.known_solutions[size] = solutions
        return solutions
    
    def _generate_candidate(self, grid: LShapeGridGPU) -> None:
        """Generate a candidate grid using various strategies"""
        strategies = [
            self._generate_diagonal,
            self._generate_cyclic,
            self._generate_random,
            self._generate_checkerboard,
            self._generate_stripes,
            self._generate_balanced_random  # New strategy
        ]
        strategy = random.choice(strategies)
        strategy(grid)
    
    def _generate_diagonal(self, grid: LShapeGridGPU) -> None:
        """Generate grid with diagonal pattern using GPU"""
        colors = torch.tensor([c.value for c in Color], device=self.device)
        colors = colors[torch.randperm(3, device=self.device)]
        
        y_coords, x_coords = torch.meshgrid(
            torch.arange(grid.size, device=self.device),
            torch.arange(grid.size, device=self.device),
            indexing='ij'
        )
        
        color_indices = (x_coords + y_coords) % 3
        grid.grid = colors[color_indices]
    
    def _generate_cyclic(self, grid: LShapeGridGPU) -> None:
        """Generate grid with cyclic pattern using GPU"""
        colors = torch.tensor([c.value for c in Color], device=self.device)
        colors = colors[torch.randperm(3, device=self.device)]
        
        y_coords, x_coords = torch.meshgrid(
            torch.arange(grid.size, device=self.device),
            torch.arange(grid.size, device=self.device),
            indexing='ij'
        )
        
        color_indices = (x_coords + 2*y_coords) % 3
        grid.grid = colors[color_indices]
    
    def _generate_random(self, grid: LShapeGridGPU) -> None:
        """Generate random grid using GPU"""
        colors = torch.tensor([c.value for c in Color], device=self.device)
        random_indices = torch.randint(0, 3, (grid.size, grid.size), device=self.device)
        grid.grid = colors[random_indices]
    
    def _generate_checkerboard(self, grid: LShapeGridGPU) -> None:
        """Generate checkerboard pattern"""
        colors = torch.tensor([c.value for c in Color], device=self.device)
        colors = colors[torch.randperm(3, device=self.device)]
        
        y_coords, x_coords = torch.meshgrid(
            torch.arange(grid.size, device=self.device),
            torch.arange(grid.size, device=self.device),
            indexing='ij'
        )
        
        color_indices = ((x_coords + y_coords) % 2) + (x_coords % 2)
        grid.grid = colors[color_indices]
    
    def _generate_stripes(self, grid: LShapeGridGPU) -> None:
        """Generate striped pattern"""
        colors = torch.tensor([c.value for c in Color], device=self.device)
        colors = colors[torch.randperm(3, device=self.device)]
        
        y_coords, x_coords = torch.meshgrid(
            torch.arange(grid.size, device=self.device),
            torch.arange(grid.size, device=self.device),
            indexing='ij'
        )
        
        color_indices = (x_coords // 2) % 3
        grid.grid = colors[color_indices]
    
    def _generate_balanced_random(self, grid: LShapeGridGPU) -> None:
        """Generate a random grid with balanced color distribution"""
        colors = torch.tensor([c.value for c in Color], device=self.device)
        # Create a balanced distribution of colors
        num_cells = grid.size * grid.size
        cells_per_color = num_cells // 3
        # Add extra cells to ensure we have enough
        color_indices = torch.cat([
            torch.full((cells_per_color + (1 if i < num_cells % 3 else 0),), i, device=self.device)
            for i in range(3)
        ])
        # Shuffle the indices
        color_indices = color_indices[torch.randperm(len(color_indices), device=self.device)]
        # Reshape to grid
        grid.grid = color_indices[:num_cells].view(grid.size, grid.size)
    
    def build_larger_grid(self, target_size: int, base_size: int = 5) -> LShapeGridGPU:
        """Build a larger grid using known valid smaller grids"""
        base_solutions = self.find_base_solutions(base_size)
        if not base_solutions:
            raise ValueError(f"No valid {base_size}x{base_size} solutions found")
        
        grid = LShapeGridGPU(target_size)
        
        for by in range(0, target_size, base_size):
            for bx in range(0, target_size, base_size):
                base = random.choice(base_solutions)
                block_width = min(base_size, target_size - bx)
                block_height = min(base_size, target_size - by)
                
                attempts = 0
                while attempts < 10:
                    # Copy block using GPU operations
                    grid.grid[by:by+block_height, bx:bx+block_width] = \
                        base.grid[:block_height, :block_width]
                    
                    # Check for L-shapes
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
                    
                    base = random.choice(base_solutions)
                    attempts += 1
                
                if attempts == 10:
                    print(f"Warning: Could not find valid arrangement for block at ({bx},{by})")
        
        return grid
    
    def grid_to_hash(self, grid: LShapeGridGPU) -> str:
        """Convert grid to a unique hash string using GPU"""
        # Convert grid to CPU for hashing
        colors = grid.grid.cpu().numpy()
        return hashlib.sha256(str(colors).encode()).hexdigest()

def evaluate_grid(grid: LShapeGridGPU) -> float:
    """
    Evaluate how good a grid configuration is using GPU acceleration.
    Returns a score where:
    - Higher is better
    - 0 means invalid (has L-shape)
    - Positive values indicate valid configurations
    """
    has_l, _ = grid.has_any_l_shape()
    if has_l:
        return 0
    
    # Count color diversity and patterns using GPU operations
    score = 0
    
    # Reward color diversity in rows and columns
    for i in range(grid.size):
        # Get unique colors in row and column using GPU operations
        row_colors = torch.unique(grid.grid[i, :])
        col_colors = torch.unique(grid.grid[:, i])
        
        # More diverse rows/columns get higher scores
        score += len(row_colors) + len(col_colors)
    
    # Penalize adjacent same colors (to discourage potential L-shapes)
    # Use GPU operations for efficient comparison
    right_diff = grid.grid[:, :-1] != grid.grid[:, 1:]
    bottom_diff = grid.grid[:-1, :] != grid.grid[1:, :]
    score -= 0.5 * (torch.sum(~right_diff) + torch.sum(~bottom_diff))
    
    return score

def mutate_grid(grid: LShapeGridGPU) -> LShapeGridGPU:
    """Create a slightly modified version of the grid using GPU operations"""
    new_grid = LShapeGridGPU(grid.size)
    new_grid.grid = grid.grid.clone()
    
    # Randomly change a few cells using GPU operations
    num_changes = random.randint(1, max(2, grid.size // 2))
    change_indices = torch.randperm(grid.size * grid.size, device=grid.device)[:num_changes]
    
    for idx in change_indices:
        y = idx // grid.size
        x = idx % grid.size
        # Avoid using the same color that's currently there
        current_color = new_grid.grid[y, x]
        available_colors = torch.tensor([c.value for c in Color if c.value != current_color], 
                                      device=grid.device)
        new_grid.grid[y, x] = available_colors[torch.randint(0, len(available_colors), (1,), device=grid.device)]
    
    return new_grid

def crossover(grid1: LShapeGridGPU, grid2: LShapeGridGPU) -> LShapeGridGPU:
    """Create a new grid by combining two parent grids using GPU operations"""
    if grid1.size != grid2.size:
        raise ValueError("Grids must be the same size")
    
    child = LShapeGridGPU(grid1.size)
    
    # Create a random mask for selection using GPU operations
    mask = torch.rand(grid1.size, grid1.size, device=grid1.device) < 0.5
    child.grid = torch.where(mask, grid1.grid, grid2.grid)
    
    return child

def funsearch_gpu(size: int, max_iterations: int = 10000, time_limit: int = 300) -> Tuple[LShapeGridGPU, float]:
    """
    Run FunSearch with GPU acceleration to find a valid grid configuration.
    Returns the best grid found and its score.
    """
    best_grid = None
    best_score = 0
    
    # Start with random grids
    population_size = 100
    population = [LShapeGridGPU(size) for _ in range(population_size)]
    
    # Initialize random grids using GPU operations
    for grid in population:
        colors = torch.tensor([c.value for c in Color], device=grid.device)
        random_indices = torch.randint(0, 3, (size, size), device=grid.device)
        grid.grid = colors[random_indices]
    
    start_time = time.time()
    iteration = 0
    
    while iteration < max_iterations and (time.time() - start_time) < time_limit:
        # Evaluate current population using GPU
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
                best_grid.visualize(filename=f"visualizations/grid_{size}x{size}_solution_{iteration}.png")
                print(f"Saved visualization to: visualizations/grid_{size}x{size}_solution_{iteration}.png")
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
    print("Starting L-shape Ramsey Grid Search with GPU-accelerated FunSearch")
    print(f"Using device: {get_device()}")
    print(f"Initial GPU Memory: {get_gpu_memory_usage():.1f}GB")
    
    # Try to find solutions for different sizes
    sizes = [4, 5, 6]
    for size in sizes:
        print(f"\n{'='*50}")
        print(f"Searching for {size}x{size} solutions")
        print(f"{'='*50}")
        best_grid, score = funsearch_gpu(size, max_iterations=10000, time_limit=300)
        
        if best_grid is not None:
            print(f"\nBest solution found for {size}x{size}:")
            print(f"Score: {score}")
            has_l, points = best_grid.has_any_l_shape()
            print(f"Has L-shape: {has_l}")
            if has_l:
                print(f"L-shape points: {points}")
            best_grid.visualize(filename=f"visualizations/grid_{size}x{size}_final.png")
            print(f"Saved visualization to: visualizations/grid_{size}x{size}_final.png")
        else:
            print(f"No valid solution found for {size}x{size}")

if __name__ == "__main__":
    main() 