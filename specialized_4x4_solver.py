#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import argparse
from l_shape_ramsey import LShapeGrid, Color

class Specialized4x4Solver:
    """
    A specialized solver for the 4×4 grid L-shape Ramsey problem.
    This solver uses custom patterns designed specifically for the 4×4 grid.
    """
    
    def __init__(self):
        """Initialize the solver."""
        # Output directory for visualizations
        self.vis_dir = "visualizations"
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # Total number of solutions to try
        self.num_solutions_to_try = 10
    
    def generate_solution_1(self):
        """
        Generate a specialized 4×4 solution using a specific pattern.
        Based on a pattern that alternates colors in a specific way.
        """
        return np.array([
            [0, 1, 2, 0],
            [2, 0, 1, 2],
            [1, 2, 0, 1],
            [0, 1, 2, 0]
        ])
    
    def generate_solution_2(self):
        """
        Generate an alternative specialized 4×4 solution.
        Using a different arrangement of colors.
        """
        return np.array([
            [0, 1, 2, 0],
            [1, 2, 0, 1],
            [2, 0, 1, 2],
            [0, 1, 2, 0]
        ])
    
    def generate_solution_3(self):
        """
        Generate another specialized 4×4 solution.
        Using a corner-focused approach.
        """
        return np.array([
            [0, 1, 1, 2],
            [1, 0, 2, 1],
            [1, 2, 0, 1],
            [2, 1, 1, 0]
        ])
    
    def generate_solution_4(self):
        """
        Generate another specialized 4×4 solution.
        Using a different corner-focused approach.
        """
        return np.array([
            [0, 1, 2, 0],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [0, 2, 1, 0]
        ])
    
    def generate_solution_5(self):
        """
        Generate a specialized 4×4 solution with a modified checkerboard pattern.
        """
        return np.array([
            [0, 1, 0, 1],
            [2, 0, 2, 0],
            [0, 2, 0, 2],
            [1, 0, 1, 0]
        ])
    
    def generate_solution_6(self):
        """
        Generate a specialized 4×4 solution with diagonal-based pattern.
        """
        return np.array([
            [0, 2, 1, 0],
            [1, 0, 2, 1],
            [2, 1, 0, 2],
            [0, 2, 1, 0]
        ])
    
    def generate_solution_7(self):
        """
        Generate a specialized 4×4 solution with diagonal-based pattern (reversed).
        """
        return np.array([
            [0, 1, 2, 0],
            [2, 0, 1, 2],
            [1, 2, 0, 1],
            [0, 1, 2, 0]
        ])
    
    def generate_solution_8(self):
        """
        Generate a specialized 4×4 solution with a spiral pattern.
        """
        return np.array([
            [0, 1, 2, 0],
            [2, 1, 0, 1],
            [0, 2, 1, 2],
            [1, 0, 2, 0]
        ])
    
    def generate_solution_9(self):
        """
        Generate a specialized 4×4 solution with a zigzag pattern.
        """
        return np.array([
            [0, 1, 2, 0],
            [2, 0, 1, 2],
            [0, 2, 0, 1],
            [1, 0, 2, 0]
        ])
    
    def generate_solution_10(self):
        """
        Generate a specialized 4×4 solution with a random (but fixed) pattern.
        """
        return np.array([
            [0, 2, 1, 0],
            [1, 0, 2, 1],
            [2, 1, 0, 2],
            [0, 2, 1, 0]
        ])
    
    def verify_solution(self, grid):
        """
        Verify that a grid doesn't contain monochromatic L-shapes.
        Returns True if valid, False otherwise, along with detailed information.
        """
        n = grid.shape[0]
        
        # Convert to LShapeGrid
        l_shape_grid = LShapeGrid(n)
        for i in range(n):
            for j in range(n):
                l_shape_grid.set_color(j, i, list(Color)[grid[i, j]])
        
        # Check for L-shapes
        has_l, points = l_shape_grid.has_any_l_shape()
        
        if has_l:
            return False, points
        else:
            return True, None
    
    def visualize_solution(self, grid, filename=None):
        """
        Visualize a grid solution.
        """
        plt.figure(figsize=(6, 6))
        cmap = plt.cm.colors.ListedColormap(['red', 'green', 'blue'])
        plt.imshow(grid, cmap=cmap, vmin=0, vmax=2)
        plt.grid(True, color='black', linewidth=1.5)
        plt.xticks(range(grid.shape[0]))
        plt.yticks(range(grid.shape[1]))
        plt.title("L-shape Ramsey 4×4 Grid Solution")
        
        if filename:
            plt.savefig(filename, dpi=150)
            plt.close()
        else:
            plt.show()
    
    def run(self, verbose=True):
        """
        Run the specialized solver to try all solutions.
        """
        print(f"Trying {self.num_solutions_to_try} specialized 4×4 solutions...")
        
        valid_solutions = []
        
        for i in range(1, self.num_solutions_to_try + 1):
            if verbose:
                print(f"\nTrying solution {i}...")
            
            # Generate solution
            solution_func = getattr(self, f"generate_solution_{i}")
            grid = solution_func()
            
            if verbose:
                print(grid)
            
            # Verify solution
            start_time = time.time()
            is_valid, points = self.verify_solution(grid)
            verification_time = time.time() - start_time
            
            if is_valid:
                if verbose:
                    print(f"✅ Solution {i} is VALID! Verification time: {verification_time:.2f}s")
                
                # Save valid solution
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.vis_dir, f"valid_4x4_solution_{i}_{timestamp}.png")
                self.visualize_solution(grid, filename)
                
                valid_solutions.append((i, grid))
            else:
                if verbose:
                    print(f"❌ Solution {i} is INVALID. Found L-shape at {points}. Verification time: {verification_time:.2f}s")
        
        # Summary
        if valid_solutions:
            print(f"\nFound {len(valid_solutions)} valid solutions!")
            for idx, (solution_num, grid) in enumerate(valid_solutions):
                print(f"\nValid Solution #{idx+1} (Solution #{solution_num}):")
                print(grid)
        else:
            print("\nNo valid solutions found.")
        
        return valid_solutions

def main():
    parser = argparse.ArgumentParser(description='Specialized 4×4 grid solver for L-shape Ramsey problem')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output')
    args = parser.parse_args()
    
    solver = Specialized4x4Solver()
    solver.run(verbose=args.verbose)

if __name__ == "__main__":
    main() 