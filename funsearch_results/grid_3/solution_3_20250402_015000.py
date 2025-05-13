# Solution for 3x3 grid
# Score: 7.0
# Generated: 20250402_015000

import numpy as np

def generate_grid(n):
    import numpy as np
    grid = np.zeros((n, n), dtype=int)
    # Latin square pattern for 3x3
    pattern = [
        [0, 1, 2],
        [2, 0, 1],
        [1, 2, 0]
    ]
    for i in range(min(n, 3)):
        for j in range(min(n, 3)):
            grid[i, j] = pattern[i][j]
    return grid
