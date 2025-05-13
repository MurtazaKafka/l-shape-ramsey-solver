#!/usr/bin/env python3
"""
Run FunSearch for the L-shape Ramsey problem with advanced settings.

This script sets up and runs FunSearch to evolve code for solving the L-shape Ramsey problem.
"""
import os
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import multiprocessing as mp

# Import our specification
from advanced_l_shape_specification import (
    generate_l_shape_ramsey_grid,
    evaluate_l_shape_ramsey,
    LShapeRamseyGrid,
    get_device
)

# Create directory for visualizations
VIZ_DIR = "visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)

# Create directory for FunSearch results
RESULTS_DIR = "funsearch_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuration
class FunSearchConfig:
    """Configuration for FunSearch run."""
    # Grid sizes to attempt
    GRID_SIZES = [3, 4, 5, 6, 7, 8, 9, 10]
    
    # Number of colors
    NUM_COLORS = 3
    
    # Number of iterations per grid size
    ITERATIONS_PER_SIZE = 500
    
    # Number of parallel processes to use
    NUM_PROCESSES = min(mp.cpu_count(), 16)
    
    # Population size for each grid size
    POPULATION_SIZE = 50
    
    # Number of LLM samples to generate per prompt
    SAMPLES_PER_PROMPT = 10
    
    # How often to save results (iterations)
    SAVE_FREQUENCY = 50
    
    # How often to visualize grids (iterations)
    VISUALIZATION_FREQUENCY = 100
    
    # Initial temperature for code evolution
    INITIAL_TEMPERATURE = 0.8
    
    # Final temperature
    FINAL_TEMPERATURE = 0.1
    
    # Time limit in seconds (4 hours)
    TIME_LIMIT = 4 * 60 * 60


def visualize_grid(grid: LShapeRamseyGrid, size: int, timestamp: str):
    """Visualize a grid and save to file."""
    plt.figure(figsize=(8, 8))
    
    # Define colors
    color_map = {
        0: 'red',
        1: 'green',
        2: 'blue',
        -1: 'white'
    }
    
    # Convert grid tensor to numpy for visualization
    grid_np = grid.grid.cpu().numpy()
    
    # Plot the grid
    for y in range(size):
        for x in range(size):
            color_idx = grid_np[y, x]
            cell_color = color_map.get(color_idx, 'gray')
            rect = plt.Rectangle((x, size - 1 - y), 1, 1, 
                                facecolor=cell_color, edgecolor='black')
            plt.gca().add_patch(rect)
    
    # Set limits and grid
    plt.xlim(0, size)
    plt.ylim(0, size)
    plt.xticks(range(size + 1))
    plt.yticks(range(size + 1))
    plt.grid(True)
    
    # Add title
    plt.title(f'L-shape Ramsey Grid ({size}x{size})')
    
    # Save the figure
    filename = f"{VIZ_DIR}/funsearch_l_shape_grid_{size}x{size}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename


def run_funsearch_for_grid_size(grid_size: int, config: FunSearchConfig) -> Dict[str, Any]:
    """Run FunSearch for a specific grid size."""
    print(f"\n{'='*50}")
    print(f"Starting FunSearch for grid size {grid_size}x{grid_size}")
    print(f"{'='*50}")
    
    # Set random seed
    base_seed = int(time.time()) % 10000
    
    # Track best results
    best_score = 0.0
    best_genotype = None
    best_grid = None
    generation = 0
    
    # Track all valid results
    valid_results = []
    
    start_time = time.time()
    
    # Initialize population of genotypes
    population = []
    population_scores = []
    
    for i in range(config.POPULATION_SIZE):
        seed = base_seed + i
        genotype = generate_l_shape_ramsey_grid(grid_size, config.NUM_COLORS, seed)
        
        # Evaluate genotype
        score = evaluate_l_shape_ramsey((grid_size, config.NUM_COLORS, seed))
        
        population.append(genotype)
        population_scores.append(score)
        
        # Update best
        if score > best_score:
            best_score = score
            best_genotype = genotype
            
            # Create grid for visualization
            grid = LShapeRamseyGrid(grid_size, config.NUM_COLORS)
            grid.fill_grid_from_genotype(genotype)
            best_grid = grid
            
            # If valid solution, add to results
            if score > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                valid_results.append({
                    "grid_size": grid_size,
                    "score": score,
                    "genotype": genotype,
                    "timestamp": timestamp,
                    "generation": generation
                })
                
                # Visualize
                visualize_grid(grid, grid_size, timestamp)
                print(f"  Generation {generation}: Found valid solution with score {score:.2f}")
    
    # Main FunSearch loop
    for iteration in range(config.ITERATIONS_PER_SIZE):
        generation += 1
        
        # Check time limit
        if time.time() - start_time > config.TIME_LIMIT:
            print(f"Time limit reached after {generation} generations")
            break
        
        # Calculate temperature (anneal from initial to final)
        progress = min(1.0, iteration / config.ITERATIONS_PER_SIZE)
        temperature = config.INITIAL_TEMPERATURE * (1 - progress) + config.FINAL_TEMPERATURE * progress
        
        # Generate new genotypes through evolution or sampling
        new_population = []
        new_scores = []
        
        # Elitism: Keep the best solutions
        elite_count = max(1, config.POPULATION_SIZE // 10)
        elite_indices = np.argsort(population_scores)[-elite_count:]
        
        for idx in elite_indices:
            new_population.append(population[idx])
            new_scores.append(population_scores[idx])
        
        # Calculate selection probabilities (softmax with temperature)
        selection_probs = np.exp(np.array(population_scores) / temperature)
        selection_probs = selection_probs / np.sum(selection_probs)
        
        # Generate new individuals
        while len(new_population) < config.POPULATION_SIZE:
            # 80% chance: Create by mutation/crossover
            if random.random() < 0.8:
                # Select parent(s)
                parent_idx = np.random.choice(len(population), p=selection_probs)
                parent = population[parent_idx]
                
                # 70% mutation, 30% crossover
                if random.random() < 0.7:
                    # Mutation: Change pattern type or parameters
                    child = parent.copy()
                    
                    # 50% chance to change pattern type
                    if random.random() < 0.5:
                        pattern_types = ["modulo", "block", "recursive", "formula", "random"]
                        child["pattern_type"] = random.choice(pattern_types)
                        
                        # Initialize parameters based on pattern type
                        if child["pattern_type"] == "modulo":
                            child["params"] = {
                                "a": random.choice([1, 2, 3]),
                                "b": random.choice([1, 2, 3]),
                                "c": random.randint(0, 5)
                            }
                        elif child["pattern_type"] == "block":
                            child["params"] = {
                                "block_size": random.choice([2, 3])
                            }
                        elif child["pattern_type"] == "formula":
                            child["params"] = {
                                "formula_type": random.randint(0, 3)
                            }
                        else:
                            child["params"] = {}
                    
                    # 50% chance to modify parameters
                    else:
                        if "params" not in child:
                            child["params"] = {}
                            
                        pattern_type = child["pattern_type"]
                        
                        if pattern_type == "modulo":
                            # Mutate modulo parameters
                            if random.random() < 0.33:
                                child["params"]["a"] = random.choice([1, 2, 3, 4, 5])
                            if random.random() < 0.33:
                                child["params"]["b"] = random.choice([1, 2, 3, 4, 5])
                            if random.random() < 0.33:
                                child["params"]["c"] = random.randint(0, 10)
                                
                        elif pattern_type == "block":
                            # Mutate block parameters
                            child["params"]["block_size"] = random.choice([2, 3, 4])
                            
                        elif pattern_type == "formula":
                            # Mutate formula parameters
                            child["params"]["formula_type"] = random.randint(0, 3)
                else:
                    # Crossover: Select second parent and combine
                    parent2_idx = np.random.choice(len(population), p=selection_probs)
                    parent2 = population[parent2_idx]
                    
                    # Basic crossover: Take pattern_type from one, params from other
                    child = {}
                    
                    # 50/50 chance for each key
                    child["pattern_type"] = parent["pattern_type"] if random.random() < 0.5 else parent2["pattern_type"]
                    
                    # Combine parameters
                    child["params"] = {}
                    p1_params = parent.get("params", {})
                    p2_params = parent2.get("params", {})
                    
                    # Get all parameter keys
                    all_keys = set(list(p1_params.keys()) + list(p2_params.keys()))
                    
                    # For each parameter, randomly choose from which parent to inherit
                    for key in all_keys:
                        if key in p1_params and key in p2_params:
                            # If both parents have this parameter, randomly choose
                            child["params"][key] = p1_params[key] if random.random() < 0.5 else p2_params[key]
                        elif key in p1_params:
                            # Only parent 1 has this parameter
                            child["params"][key] = p1_params[key]
                        else:
                            # Only parent 2 has this parameter
                            child["params"][key] = p2_params[key]
                
                # Generate a new seed
                seed = base_seed + random.randint(0, 10000)
                
                # Evaluate the child
                score = evaluate_l_shape_ramsey((grid_size, config.NUM_COLORS, seed))
                
                new_population.append(child)
                new_scores.append(score)
            
            # 20% chance: Generate completely new individual
            else:
                seed = base_seed + random.randint(0, 10000)
                new_genotype = generate_l_shape_ramsey_grid(grid_size, config.NUM_COLORS, seed)
                score = evaluate_l_shape_ramsey((grid_size, config.NUM_COLORS, seed))
                
                new_population.append(new_genotype)
                new_scores.append(score)
        
        # Update population
        population = new_population
        population_scores = new_scores
        
        # Update best if improved
        current_best_idx = np.argmax(population_scores)
        current_best_score = population_scores[current_best_idx]
        
        if current_best_score > best_score:
            best_score = current_best_score
            best_genotype = population[current_best_idx]
            
            # Create grid for visualization
            grid = LShapeRamseyGrid(grid_size, config.NUM_COLORS)
            grid.fill_grid_from_genotype(best_genotype)
            best_grid = grid
            
            # If valid solution, add to results
            if best_score > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                valid_results.append({
                    "grid_size": grid_size,
                    "score": best_score,
                    "genotype": best_genotype,
                    "timestamp": timestamp,
                    "generation": generation
                })
                
                # Visualize
                viz_file = visualize_grid(grid, grid_size, timestamp)
                print(f"  Generation {generation}: New best solution with score {best_score:.2f}")
                print(f"  Visualization saved to {viz_file}")
        
        # Periodic status update
        if generation % 10 == 0:
            print(f"  Generation {generation}: Best score = {best_score:.2f}, Avg score = {np.mean(population_scores):.2f}")
        
        # Save results periodically
        if generation % config.SAVE_FREQUENCY == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"{RESULTS_DIR}/funsearch_results_{grid_size}x{grid_size}_{timestamp}.txt"
            
            with open(results_file, "w") as f:
                f.write(f"Grid Size: {grid_size}x{grid_size}\n")
                f.write(f"Generation: {generation}\n")
                f.write(f"Best Score: {best_score}\n")
                f.write(f"Valid Solutions Found: {len(valid_results)}\n\n")
                
                for i, result in enumerate(valid_results):
                    f.write(f"Solution {i+1}:\n")
                    f.write(f"  Score: {result['score']}\n")
                    f.write(f"  Generation: {result['generation']}\n")
                    f.write(f"  Timestamp: {result['timestamp']}\n")
                    f.write(f"  Genotype: {result['genotype']}\n\n")
    
    # Final results
    print(f"\nFunSearch completed for grid size {grid_size}x{grid_size}")
    print(f"Total generations: {generation}")
    print(f"Best score: {best_score}")
    print(f"Valid solutions found: {len(valid_results)}")
    
    # Return best result
    return {
        "grid_size": grid_size,
        "best_score": best_score,
        "best_genotype": best_genotype,
        "best_grid": best_grid,
        "valid_solutions": valid_results,
        "generations": generation
    }


def run_funsearch_worker(grid_size: int, config: FunSearchConfig, results_queue):
    """Worker function for parallel processing."""
    try:
        result = run_funsearch_for_grid_size(grid_size, config)
        results_queue.put(result)
    except Exception as e:
        print(f"Error in worker for grid size {grid_size}: {e}")
        results_queue.put({"grid_size": grid_size, "error": str(e)})


def run_full_funsearch():
    """Run FunSearch for all grid sizes in parallel."""
    config = FunSearchConfig()
    print(f"Starting FunSearch for L-shape Ramsey Problem")
    print(f"Grid sizes: {config.GRID_SIZES}")
    print(f"Number of colors: {config.NUM_COLORS}")
    print(f"Using {config.NUM_PROCESSES} parallel processes")
    print(f"Time limit: {config.TIME_LIMIT} seconds")
    
    start_time = time.time()
    
    # Create a queue for results
    results_queue = mp.Queue()
    
    # Launch processes
    processes = []
    
    # Initialize with smaller grids first
    for grid_size in config.GRID_SIZES:
        p = mp.Process(target=run_funsearch_worker, args=(grid_size, config, results_queue))
        processes.append(p)
        p.start()
        
        # Limit concurrent processes
        if len(processes) >= config.NUM_PROCESSES:
            # Wait for some process to finish before starting more
            for p in processes:
                if not p.is_alive():
                    processes.remove(p)
            
            # If we're still at max processes, wait a bit
            if len(processes) >= config.NUM_PROCESSES:
                time.sleep(1.0)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Collect results
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    
    # Sort results by grid size
    results.sort(key=lambda x: x.get("grid_size", 0))
    
    # Print summary
    print("\n" + "="*50)
    print("FunSearch Results Summary:")
    print("="*50)
    
    for result in results:
        grid_size = result.get("grid_size")
        
        if "error" in result:
            print(f"Grid {grid_size}x{grid_size}: Error - {result['error']}")
            continue
            
        best_score = result.get("best_score", 0)
        valid_solutions = result.get("valid_solutions", [])
        generations = result.get("generations", 0)
        
        valid_text = "Valid solution" if best_score > 0 else "No valid solution"
        print(f"Grid {grid_size}x{grid_size}: {valid_text}, Score: {best_score:.2f}, Generations: {generations}")
    
    # Save final visualizations for best results
    print("\nGenerating final visualizations...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for result in results:
        if "best_grid" in result and result["best_grid"] is not None:
            grid_size = result["grid_size"]
            grid = result["best_grid"]
            visualize_grid(grid, grid_size, f"final_{timestamp}")
    
    # Save comprehensive results
    final_results_file = f"{RESULTS_DIR}/funsearch_final_results_{timestamp}.txt"
    with open(final_results_file, "w") as f:
        f.write(f"FunSearch for L-shape Ramsey Problem\n")
        f.write(f"Run completed at: {datetime.now()}\n")
        f.write(f"Total time: {(time.time() - start_time):.2f} seconds\n\n")
        
        for result in results:
            grid_size = result.get("grid_size")
            
            if "error" in result:
                f.write(f"Grid {grid_size}x{grid_size}: Error - {result['error']}\n\n")
                continue
                
            best_score = result.get("best_score", 0)
            valid_solutions = result.get("valid_solutions", [])
            generations = result.get("generations", 0)
            best_genotype = result.get("best_genotype")
            
            f.write(f"Grid {grid_size}x{grid_size}:\n")
            f.write(f"  Best Score: {best_score}\n")
            f.write(f"  Generations: {generations}\n")
            f.write(f"  Valid Solutions: {len(valid_solutions)}\n")
            f.write(f"  Best Genotype: {best_genotype}\n\n")
    
    print(f"\nFinal results saved to {final_results_file}")
    print(f"\nTotal time: {(time.time() - start_time):.2f} seconds")


if __name__ == "__main__":
    mp.set_start_method('spawn')  # Needed for CUDA/MPS compatibility
    run_full_funsearch() 