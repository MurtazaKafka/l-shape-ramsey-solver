#!/usr/bin/env python3
"""
Analyze FunSearch results for the L-shape Ramsey problem.

This script loads and analyzes results from previous FunSearch runs, generating
visualizations and statistics about the discovered solutions.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from datetime import datetime
import re
import ast

from advanced_l_shape_specification import LShapeRamseyGrid, get_device

# Directories
RESULTS_DIR = "funsearch_results"
VIZ_DIR = "visualizations"
ANALYSIS_DIR = "analysis"

# Create analysis directory
os.makedirs(ANALYSIS_DIR, exist_ok=True)


def parse_genotype_str(genotype_str):
    """Parse genotype string into a Python dictionary."""
    try:
        # Remove any trailing commas in dictionaries
        cleaned_str = re.sub(r',\s*}', '}', genotype_str)
        return ast.literal_eval(cleaned_str)
    except:
        # Fallback to a default genotype if parsing fails
        return {
            "pattern_type": "modulo",
            "params": {
                "a": 1,
                "b": 2,
                "c": 0
            }
        }


def visualize_grid_from_genotype(genotype, grid_size, num_colors=3, filename=None):
    """Generate and visualize a grid from a genotype."""
    # Create grid
    grid = LShapeRamseyGrid(grid_size, num_colors)
    grid.fill_grid_from_genotype(genotype)
    
    # Visualize
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
    for y in range(grid_size):
        for x in range(grid_size):
            color_idx = grid_np[y, x]
            cell_color = color_map.get(color_idx, 'gray')
            rect = plt.Rectangle((x, grid_size - 1 - y), 1, 1, 
                                facecolor=cell_color, edgecolor='black')
            plt.gca().add_patch(rect)
    
    # Set limits and grid
    plt.xlim(0, grid_size)
    plt.ylim(0, grid_size)
    plt.xticks(range(grid_size + 1))
    plt.yticks(range(grid_size + 1))
    plt.grid(True)
    
    # Check for L-shapes
    has_l_shape = grid.has_l_shape()
    l_status = "Invalid (L-shapes present)" if has_l_shape else "Valid (No L-shapes)"
    
    # Add title
    plt.title(f'L-shape Ramsey Grid ({grid_size}x{grid_size}) - {l_status}')
    
    # Save or display
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return grid, has_l_shape
    else:
        plt.show()
        return grid, has_l_shape


def parse_results_file(filepath):
    """Parse a results file and extract solution data."""
    solutions = []
    
    with open(filepath, 'r') as f:
        content = f.read()
        
    # Extract grid size
    grid_size_match = re.search(r'Grid Size: (\d+)x(\d+)', content)
    if grid_size_match:
        grid_size = int(grid_size_match.group(1))
    else:
        grid_size = None
    
    # Extract solutions
    solution_blocks = re.findall(r'Solution \d+:(.*?)(?=Solution \d+:|$)', content, re.DOTALL)
    
    for block in solution_blocks:
        solution = {}
        
        # Extract score
        score_match = re.search(r'Score: ([\d\.]+)', block)
        if score_match:
            solution['score'] = float(score_match.group(1))
        
        # Extract generation
        gen_match = re.search(r'Generation: (\d+)', block)
        if gen_match:
            solution['generation'] = int(gen_match.group(1))
        
        # Extract timestamp
        time_match = re.search(r'Timestamp: (\d+)', block)
        if time_match:
            solution['timestamp'] = time_match.group(1)
        
        # Extract genotype
        genotype_match = re.search(r'Genotype: ({.*})', block, re.DOTALL)
        if genotype_match:
            genotype_str = genotype_match.group(1)
            solution['genotype'] = parse_genotype_str(genotype_str)
        
        if solution and 'genotype' in solution:
            solution['grid_size'] = grid_size
            solutions.append(solution)
    
    return solutions


def find_all_result_files():
    """Find all result files in the results directory."""
    return glob.glob(f"{RESULTS_DIR}/funsearch_results_*.txt")


def analyze_results():
    """Analyze all result files and generate visualizations."""
    print("Analyzing FunSearch results...")
    
    # Find all result files
    result_files = find_all_result_files()
    print(f"Found {len(result_files)} result files.")
    
    if not result_files:
        print("No result files found. Run FunSearch first.")
        return
    
    # Parse all solutions
    all_solutions = []
    for filepath in result_files:
        solutions = parse_results_file(filepath)
        all_solutions.extend(solutions)
    
    print(f"Parsed {len(all_solutions)} solutions in total.")
    
    # Group solutions by grid size
    solutions_by_size = {}
    for solution in all_solutions:
        grid_size = solution.get('grid_size')
        if grid_size:
            if grid_size not in solutions_by_size:
                solutions_by_size[grid_size] = []
            solutions_by_size[grid_size].append(solution)
    
    # Sort grid sizes
    grid_sizes = sorted(solutions_by_size.keys())
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # For each grid size, find the best solution and visualize it
    best_solutions = {}
    print("\nGenerating visualizations for best solutions:")
    
    for grid_size in grid_sizes:
        solutions = solutions_by_size[grid_size]
        
        # Sort by score (higher is better)
        solutions.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Take the best solution
        best_solution = solutions[0]
        best_solutions[grid_size] = best_solution
        
        print(f"Grid {grid_size}x{grid_size}: Best score = {best_solution['score']}")
        
        # Visualize
        if 'genotype' in best_solution:
            viz_file = f"{ANALYSIS_DIR}/best_grid_{grid_size}x{grid_size}_{timestamp}.png"
            grid, has_l = visualize_grid_from_genotype(
                best_solution['genotype'], grid_size, filename=viz_file)
            
            valid_text = "valid" if not has_l else "invalid"
            print(f"  Generated {valid_text} visualization at {viz_file}")
    
    # Generate comparative visualizations
    
    # 1. Grid size vs best score
    plt.figure(figsize=(10, 6))
    sizes = []
    scores = []
    valid_sizes = []
    valid_scores = []
    invalid_sizes = []
    invalid_scores = []
    
    for grid_size, solution in best_solutions.items():
        score = solution.get('score', 0)
        sizes.append(grid_size)
        scores.append(score)
        
        # Check if the solution is valid
        if score > 0:
            valid_sizes.append(grid_size)
            valid_scores.append(score)
        else:
            invalid_sizes.append(grid_size)
            invalid_scores.append(score)
    
    plt.scatter(valid_sizes, valid_scores, c='green', marker='o', s=100, label='Valid Solutions')
    plt.scatter(invalid_sizes, invalid_scores, c='red', marker='x', s=100, label='Invalid Solutions')
    
    plt.xlabel('Grid Size')
    plt.ylabel('Best Score')
    plt.title('Best Score by Grid Size')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(f"{ANALYSIS_DIR}/score_by_grid_size_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Pattern type distribution
    pattern_counts = {}
    for size, solution in best_solutions.items():
        genotype = solution.get('genotype', {})
        pattern_type = genotype.get('pattern_type', 'unknown')
        
        if pattern_type not in pattern_counts:
            pattern_counts[pattern_type] = 0
        pattern_counts[pattern_type] += 1
    
    # Plot pattern distribution
    plt.figure(figsize=(10, 6))
    patterns = list(pattern_counts.keys())
    counts = [pattern_counts[p] for p in patterns]
    
    plt.bar(patterns, counts)
    plt.xlabel('Pattern Type')
    plt.ylabel('Count')
    plt.title('Pattern Type Distribution Among Best Solutions')
    plt.savefig(f"{ANALYSIS_DIR}/pattern_distribution_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Generation vs. Score scatter plot
    # This shows how score evolves with generations
    plt.figure(figsize=(10, 6))
    
    for grid_size, solutions in solutions_by_size.items():
        # Extract generation and score
        generations = [s.get('generation', 0) for s in solutions]
        scores = [s.get('score', 0) for s in solutions]
        
        # Plot
        plt.scatter(generations, scores, alpha=0.5, label=f'{grid_size}x{grid_size}')
    
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title('Score Evolution Across Generations')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{ANALYSIS_DIR}/score_evolution_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save analysis summary
    with open(f"{ANALYSIS_DIR}/analysis_summary_{timestamp}.txt", "w") as f:
        f.write(f"L-shape Ramsey Problem - FunSearch Analysis\n")
        f.write(f"Analysis completed at: {datetime.now()}\n\n")
        
        f.write("Grid Size Summary:\n")
        for grid_size in grid_sizes:
            solutions = solutions_by_size[grid_size]
            valid_count = sum(1 for s in solutions if s.get('score', 0) > 0)
            
            best_solution = best_solutions[grid_size]
            score = best_solution.get('score', 0)
            valid_text = "Valid solution" if score > 0 else "No valid solution"
            
            f.write(f"Grid {grid_size}x{grid_size}: {valid_text}\n")
            f.write(f"  Best Score: {score}\n")
            f.write(f"  Valid Solutions Found: {valid_count} out of {len(solutions)}\n")
            
            # If valid, include pattern information
            if score > 0:
                genotype = best_solution.get('genotype', {})
                pattern_type = genotype.get('pattern_type', 'unknown')
                params = genotype.get('params', {})
                
                f.write(f"  Pattern Type: {pattern_type}\n")
                f.write(f"  Parameters: {params}\n")
            
            f.write("\n")
        
        f.write("\nPattern Type Distribution:\n")
        for pattern, count in pattern_counts.items():
            f.write(f"  {pattern}: {count}\n")
    
    print(f"\nAnalysis summary saved to {ANALYSIS_DIR}/analysis_summary_{timestamp}.txt")
    

def generate_multiple_grid_comparison(grid_sizes=[3, 4, 5, 6, 7, 8]):
    """Generate a single figure with all grid sizes for direct comparison."""
    print("\nGenerating combined visualization of all grid sizes...")
    
    # Number of grid sizes
    n = len(grid_sizes)
    
    # Calculate grid layout
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    plt.figure(figsize=(cols*4, rows*4))
    
    # Find the best solutions
    result_files = find_all_result_files()
    
    # Parse all solutions
    all_solutions = []
    for filepath in result_files:
        solutions = parse_results_file(filepath)
        all_solutions.extend(solutions)
    
    # Group solutions by grid size
    solutions_by_size = {}
    for solution in all_solutions:
        grid_size = solution.get('grid_size')
        if grid_size:
            if grid_size not in solutions_by_size:
                solutions_by_size[grid_size] = []
            solutions_by_size[grid_size].append(solution)
    
    # Color map
    color_map = {
        0: 'red',
        1: 'green',
        2: 'blue',
        -1: 'white'
    }
    
    # For each grid size
    for i, size in enumerate(grid_sizes):
        plt.subplot(rows, cols, i+1)
        
        if size in solutions_by_size:
            # Sort by score
            solutions = sorted(solutions_by_size[size], key=lambda x: x.get('score', 0), reverse=True)
            
            if solutions:
                best_solution = solutions[0]
                genotype = best_solution.get('genotype')
                
                if genotype:
                    # Create grid
                    grid = LShapeRamseyGrid(size, 3)
                    grid.fill_grid_from_genotype(genotype)
                    
                    # Check for L-shapes
                    has_l_shape = grid.has_l_shape()
                    
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
                    plt.title(f'{size}x{size} Grid - ' + ('Valid' if not has_l_shape else 'Invalid'))
                    plt.xticks([])
                    plt.yticks([])
                    plt.grid(True)
                    continue
        
        # If we didn't find a solution or couldn't process it
        plt.text(0.5, 0.5, f"No solution for {size}x{size}", 
                 horizontalalignment='center', verticalalignment='center')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ANALYSIS_DIR}/all_grids_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined visualization saved to {filename}")


if __name__ == "__main__":
    # Analyze results and generate visualizations
    analyze_results()
    
    # Generate combined visualization
    generate_multiple_grid_comparison() 