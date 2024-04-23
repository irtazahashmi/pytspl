import networkx as nx
import numpy as np
import pandas as pd

from sclibrary.simplicial_complex.extended_graph import ExtendedGraph

"""Module for reading graph network data."""


def read_csv(
    filename: str,
    delimeter: str,
    src_col: str,
    dest_col: str,
    feature_cols: list = None,
) -> ExtendedGraph:
    """Read a csv file and returns a graph.

    Args:
        filename (str): The name of the csv file.
        delimeter (str): The delimeter used in the csv file.
        src_col (str): The name of the column containing the source nodes.
        dest_col (str): The name of the column containing the destination
        nodes.
        feature_cols (list, optional): The names of the feature columns.
        Defaults to None.

    Returns:
        ExtendedGraph: The graph read from the csv file.
    """
    df = pd.read_csv(filename, sep=delimeter)

    # Create a graph
    G = nx.Graph()
    # add edges
    for _, row in df.iterrows():
        G.add_edge(row[src_col], row[dest_col])

    # add features if any
    if feature_cols:
        for col in feature_cols:
            for _, row in df.iterrows():
                G[row[src_col]][row[dest_col]][col] = row[col]

    return ExtendedGraph(G)


def read_tntp(
    filename: str,
    src_col: str,
    dest_col: str,
    skip_rows: int,
    delimeter: str = "\t",
) -> ExtendedGraph:
    """Read a tntp file and returns a graph.

    Args:
        filename (str): The name of the tntp file.
        src_col (str): The name of the column containing the source nodes.
        dest_col (str): The name of the column containing the destination
        nodes.
        skip_rows (int): The number of (metadata) rows to skip in the tntp
        file.
        delimeter (str): The delimeter used in the tntp file. Defaults to next
        line.

    Returns:
        ExtendedGraph: The graph read from the tntp file.
    """
    # Read the file
    df = pd.read_csv(filename, skiprows=skip_rows, sep=delimeter)
    # trimmed cols names
    df.columns = [s.strip() for s in df.columns]

    # And drop the silly first andlast columns
    df.drop(["~", ";"], axis=1, inplace=True)

    # Create a graph
    G = nx.Graph()
    # add edges
    for _, row in df.iterrows():
        G.add_edge(row[src_col], row[dest_col])

    # extract features if any
    feature_cols = [
        col for col in df.columns if col not in [src_col, dest_col]
    ]

    # add features if any
    if len(feature_cols) > 0:
        for col in feature_cols:
            for _, row in df.iterrows():
                G[row[src_col]][row[dest_col]][col] = row[col]

    return ExtendedGraph(G)


def read_incidence_matrix(B1_filename: str) -> ExtendedGraph:
    """
    Read the B1 and B2 incidence matrix files.

    Args:
        B1_filename (str): The name of the B1 incidence matrix file.
        B2_filename (str): The name of the B2 incidence matrix file.

    Returns:
        ExtendedGraph: The graph read from the incidence matrix files.
    """
    B1 = pd.read_csv(B1_filename, header=None).values

    # create adjacency matrix
    nodes = B1.shape[0]
    edges = B1.shape[1]
    assert edges > 0
    assert nodes > 0

    adjacency = [[0] * nodes for _ in range(nodes)]

    for edge in range(edges):
        a, b = -1, -1
        node = 0

        while node < nodes and a == -1:
            if B1[node][edge] != 0:
                a = node
            node += 1

        while node < nodes and b == -1:
            if B1[node][edge] != 0:
                b = node
            node += 1

        if b == -1:
            b = a

        adjacency[a][b] = -1
        adjacency[b][a] = 1

    # create graph
    G = nx.from_numpy_array(np.array(adjacency))
    return ExtendedGraph(G)


def get_coordinates(
    filename: str,
    node_id_col: str,
    x_col: str,
    y_col: str,
    delimeter: str,
) -> dict:
    """
    Read a csv file and returns a dictionary of coordinates.

    Args:
        filename (str): The name of the file.
        node_id_col (str): The name of the column containing the node ids.
        x_col (str): The name of the column containing the x coordinates.
        y_col (str): The name of the column containing the y coordinates.
        delimeter (str, optional): The delimeter used in the csv file.

    Returns:
        dict: A dictionary of coordinates (node_id : (x, y)).
    """
    df_coords = pd.read_csv(filename, sep=delimeter)

    df_coords.columns = [s.strip() for s in df_coords.columns]

    # create a dictionary of coordinates (node_id : (x, y))
    return dict(
        zip(
            df_coords[node_id_col],
            zip(df_coords[x_col], df_coords[y_col]),
        )
    )
