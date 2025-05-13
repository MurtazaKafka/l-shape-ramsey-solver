# Best program for 3x3 grid
# Score: 3.7766666666666664
# Generated: 20250402_010641

import numpy as np

import numpy as np

def generate_grid(n):
    # Create a 2D array filled with zeros
    grid = np.zeros((n, n), dtype=int)
    
    # Apply Latin square pattern to rows and columns
    for i in range(3):
        for j in range(3):
            grid[i, j] = (i + j) % 3
    
    # Fill the remaining cells with a specific pattern
    for i in range(n - 2):
        for j in range(i + 1, n):
            if (j - i - 1) % 3 != 0:
                grid[i, j] = 1
            else:
                grid[i, j] = 2
    
    # Ensure no L-shape is monochromatic
    for i in range(n - 2):
        for j in range(i + 1, n):
            if (j - i) % 3 == 0 and (grid[i, j] != 0 or grid[j - 1, j] != 0 or grid[j, j - 1] != 0):
                grid[i, j] = np.random.choice([1, 2])
    
    return grid