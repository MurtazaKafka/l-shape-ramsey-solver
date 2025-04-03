# Solution for 3x3 grid
# Score: 7.0
# Generated: 20250402_013252

import numpy as np

def generate_grid(n):
    import numpy as np
    grid = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            grid[i, j] = (i + j) % 3
    return grid
