import sys
import os
import random
from itertools import combinations
import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(this_dir, '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from implementation.funsearch import evolve, run

# Parameters for the Ramsey experiment.
N = 6           # Number of vertices in the graph.
CLIQUE_SIZE = 3 # Size of the clique to avoid.

def create_random_graph(n: int):
    """
    Creates a symmetric random graph represented as an adjacency matrix.
    The edges are colored randomly with two colors: 0 and 1.
    """
    matrix = [[None] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = None  # No self-loops.
        for j in range(i + 1, n):
            color = random.choice([0, 1])
            matrix[i][j] = color
            matrix[j][i] = color
    return matrix

def count_monochromatic_cliques(matrix, clique_size: int) -> int:
    """
    Counts and returns the number of monochromatic cliques of the specified
    size in the graph.
    """
    cliques = get_monochromatic_cliques(matrix, clique_size)
    return len(cliques)

def get_monochromatic_cliques(matrix, clique_size: int):
    """
    Returns a list of monochromatic cliques (as tuples of vertices)
    in the graph.
    """
    n = len(matrix)
    cliques = []
    for vertices in combinations(range(n), clique_size):
        colors = [matrix[i][j] for i, j in combinations(vertices, 2)]
        if len(set(colors)) == 1:
            cliques.append(vertices)
    return cliques

@evolve
def ramsey_evolve():
    """
    Evolve function: generates a candidate graph represented as an adjacency matrix.
    """
    candidate_graph = create_random_graph(N)
    return candidate_graph

@run
def ramsey_run(candidate):
    """
    Run function: evaluates the candidate graph.
    Fitness is defined as the negative number of monochromatic cliques.
    """
    num_cliques = count_monochromatic_cliques(candidate, CLIQUE_SIZE)
    fitness = -num_cliques
    print(f"Candidate fitness: {fitness} (monochromatic cliques: {num_cliques})")
    return fitness