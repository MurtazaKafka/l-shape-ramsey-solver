#!/usr/bin/env python3
import os
import time
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def create_header(title):
    """Create a pretty header for console output."""
    width = len(title) + 10
    return f"\n{'=' * width}\n{' ' * 5}{title}{' ' * 5}\n{'=' * width}"

def run_simple_solver(grid_sizes, verbose=True):
    """Run the simple solver for specified grid sizes."""
    print(create_header("Simple Deterministic Solver"))
    
    cmd = ["python", "simple_solver.py", "--grid-sizes"] + [str(n) for n in grid_sizes]
    
    if verbose:
        start_time = time.time()
        subprocess.run(cmd)
        elapsed_time = time.time() - start_time
        print(f"\nSimple solver completed in {elapsed_time:.2f} seconds")
    else:
        print(f"Command: {' '.join(cmd)}")
        print("Running silently... Check visualizations directory for results")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_specialized_4x4_solver(verbose=True):
    """Run the specialized 4x4 solver."""
    print(create_header("Specialized 4×4 Solver"))
    
    cmd = ["python", "specialized_4x4_solver.py"]
    if verbose:
        cmd.append("--verbose")
    
    if verbose:
        start_time = time.time()
        subprocess.run(cmd)
        elapsed_time = time.time() - start_time
        print(f"\nSpecialized 4×4 solver completed in {elapsed_time:.2f} seconds")
    else:
        print(f"Command: {' '.join(cmd)}")
        print("Running silently... Check visualizations directory for results")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_funsearch_solver(grid_size=3, iterations=5, time_limit=120, verbose=True):
    """Run the FunSearch solver."""
    print(create_header(f"FunSearch Solver for {grid_size}×{grid_size} Grid"))
    
    cmd = ["python", "final_llama_funsearch.py", 
           "--iterations", str(iterations),
           "--time-limit", str(time_limit)]
    
    if verbose:
        start_time = time.time()
        subprocess.run(cmd)
        elapsed_time = time.time() - start_time
        print(f"\nFunSearch solver completed in {elapsed_time:.2f} seconds")
    else:
        print(f"Command: {' '.join(cmd)}")
        print("Running silently... Check funsearch_results directory for results")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def generate_summary_visualization():
    """Generate a summary visualization of our results."""
    print(create_header("Generating Summary Visualization"))
    
    # Define our known solutions
    solution_3x3 = np.array([
        [0, 1, 2],
        [2, 0, 1],
        [1, 2, 0]
    ])
    
    solution_4x4 = np.array([
        [0, 1, 1, 2],
        [1, 0, 2, 1],
        [1, 2, 0, 1],
        [2, 1, 1, 0]
    ])
    
    # Create visualizations directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    
    # Set up the figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot 3×3 solution
    cmap = plt.cm.colors.ListedColormap(['red', 'green', 'blue'])
    axs[0].imshow(solution_3x3, cmap=cmap, vmin=0, vmax=2)
    axs[0].grid(True, color='black', linewidth=1.5)
    axs[0].set_xticks(range(3))
    axs[0].set_yticks(range(3))
    axs[0].set_title("Valid 3×3 Grid Solution (Latin Square)")
    
    # Plot 4×4 solution
    axs[1].imshow(solution_4x4, cmap=cmap, vmin=0, vmax=2)
    axs[1].grid(True, color='black', linewidth=1.5)
    axs[1].set_xticks(range(4))
    axs[1].set_yticks(range(4))
    axs[1].set_title("Valid 4×4 Grid Solution (Corner-Focused)")
    
    plt.suptitle("L-shape Ramsey Problem: Valid Solutions", fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("visualizations", f"summary_visualization_{timestamp}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Summary visualization saved to: {filename}")

def print_summary():
    """Print a summary of our findings."""
    print(create_header("Summary of Findings"))
    
    print("""
Our investigation of the L-shape Ramsey problem yielded these key findings:

1. 3×3 Grid
   ✅ Successfully solved with Latin square pattern
   ✅ Successfully solved with FunSearch using Llama 3.2

2. 4×4 Grid
   ❌ Failed with simple patterns and modular arithmetic
   ✅ Successfully solved with specialized corner-focused pattern

3. 5×5 Grid and Larger
   ❌ No valid solutions found with any of our approaches
   ❓ May require more than 3 colors or more complex patterns

4. General Observations
   - Pattern complexity increases rapidly with grid size
   - No single formula works across all grid sizes
   - Specialized patterns work better than generic ones
   - FunSearch shows promise for exploring pattern space
    """)

def main():
    parser = argparse.ArgumentParser(description='Run all L-shape Ramsey solvers')
    parser.add_argument('--all', action='store_true', help='Run all solvers')
    parser.add_argument('--simple', action='store_true', help='Run simple deterministic solver')
    parser.add_argument('--specialized', action='store_true', help='Run specialized 4×4 solver')
    parser.add_argument('--funsearch', action='store_true', help='Run FunSearch solver')
    parser.add_argument('--grid-sizes', type=int, nargs='+', default=[3, 4, 5], 
                      help='Grid sizes for simple solver (default: 3 4 5)')
    parser.add_argument('--iterations', type=int, default=5,
                      help='Number of iterations for FunSearch (default: 5)')
    parser.add_argument('--time-limit', type=int, default=120,
                      help='Time limit in seconds for FunSearch (default: 120)')
    parser.add_argument('--summary', action='store_true', help='Generate summary visualization')
    parser.add_argument('--silent', action='store_true', help='Run solvers silently')
    
    args = parser.parse_args()
    
    # If no specific solvers are selected, run the summary only
    if not (args.all or args.simple or args.specialized or args.funsearch or args.summary):
        print_summary()
        generate_summary_visualization()
        return
    
    # Run selected solvers
    if args.all or args.simple:
        run_simple_solver(args.grid_sizes, verbose=not args.silent)
    
    if args.all or args.specialized:
        run_specialized_4x4_solver(verbose=not args.silent)
    
    if args.all or args.funsearch:
        run_funsearch_solver(grid_size=3, iterations=args.iterations, 
                           time_limit=args.time_limit, verbose=not args.silent)
    
    # Generate summary
    if args.all or args.summary:
        print_summary()
        generate_summary_visualization()

if __name__ == "__main__":
    main() 