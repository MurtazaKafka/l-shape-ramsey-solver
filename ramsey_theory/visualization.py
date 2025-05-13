import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys, os

this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(this_dir, '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ramsey_theory.monochomatic import create_random_graph, get_monochromatic_cliques, N, CLIQUE_SIZE

def draw_graph(matrix):
    n = len(matrix)
    G = nx.Graph()

    # Add nodes.
    G.add_nodes_from(range(n))

    # Add edges with attribute 'color'.
    for i in range(n):
        for j in range(i+1, n):
            if matrix[i][j] is not None:
                G.add_edge(i, j, color=matrix[i][j])
    
    # Get colors.
    edge_colors = ['blue' if G[u][v]['color'] == 0 else 'red' for u, v in G.edges()]

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_color='lightgreen', node_size=600)
    plt.title("Graph with edges colored (blue=0, red=1)")
    plt.show()

def main():
    graph = create_random_graph(N)
    print("Adjacency Matrix:")
    print(np.array(graph))
    
    cliques = get_monochromatic_cliques(graph, CLIQUE_SIZE)
    print(f"\nMonochromatic cliques (size = {CLIQUE_SIZE}):")
    for clique in cliques:
        print(clique)
    
    draw_graph(graph)

if __name__ == '__main__':
    main()