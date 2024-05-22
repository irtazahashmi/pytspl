import networkx as nx

from sclibrary.simplicial_complex.scbuilder import SCBuilder


def generate_random_simplicial_complex(
    num_of_nodes: int, p: float, dist_threshold: float, seed: int
) -> tuple:
    """
    Generate a random simplicial complex.

    Args:
        num_of_nodes (int): Number of nodes in the graph.
        p (float): Probability of edge creation.
        dist_threshold (float): Threshold for simplicial complex construction.
        seed (int): Seed for random number generator.

    Returns:
        tuple: Tuple of the simplicial complex and the coordinates of the
        nodes.
    """
    G = nx.erdos_renyi_graph(n=num_of_nodes, p=p, seed=seed, directed=False)

    # get random weights
    import random

    weights = [random.random() for i in range(G.number_of_edges())]
    # set the weights
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]["distance"] = weights[i]

    sc = SCBuilder(G).to_simplicial_complex(
        condition="distance", dist_threshold=dist_threshold
    )
    coordinates = nx.spring_layout(G)

    return sc, coordinates
