import os

import networkx as nx
import numpy as np
import pandas as pd

from sclibrary.simplicial_complex.scbuilder import SCBuilder

"""Module for reading simplicial complex network data."""


def read_tntp(
    filename: str,
    src_col: str,
    dest_col: str,
    skip_rows: int,
    delimeter: str = "\t",
) -> SCBuilder:
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

    return SCBuilder(G)


def read_csv(
    filename: str,
    delimeter: str,
    src_col: str,
    dest_col: str,
    feature_cols: list = None,
) -> SCBuilder:
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

    return SCBuilder(G)


def read_B2(B2_filename: str, edges: np.ndarray) -> list:
    """
    Extract triangles from the B2 incidence matrix.

    Args:
        B2_filename (str): The name of the B2 incidence matrix
        file.
        edges (np.ndarray): The edges of the graph.

    Returns:
        list: List of triangles.
    """
    B2 = pd.read_csv(B2_filename, header=None).to_numpy().T
    num_triangles = B2.shape[1]

    triangles = []
    for j in range(num_triangles):
        # Check each column of B2 for triangles
        col = B2[:, j]
        ones = np.where(col != 0)[0]
        triangle = edges[ones]
        triangle = tuple(set(triangle.flatten()))
        triangle = tuple(sorted(triangle))
        triangles.append(triangle)

    return triangles


def read_B1(B1_filename: str) -> SCBuilder:
    """
    Read the B1 incidence matrix file.

    Args:
        B1_filename (str): The name of the B1 incidence matrix file.

    Returns:
        ExtendedGraph: The graph read from the csv file.
    """
    B1 = pd.read_csv(B1_filename, header=None).to_numpy()

    # create adjacency matrix
    adjacency_mat = np.zeros((B1.shape[0], B1.shape[0]))

    for col in range(B1.shape[1]):
        col_nozero = np.where(B1[:, col] != 0)[0]
        from_node, to_node = col_nozero[0], col_nozero[1]
        adjacency_mat[from_node, to_node] = 1
        adjacency_mat[to_node, from_node] = 1

    # create graph from adjacency matrix
    g = nx.from_numpy_array(adjacency_mat)
    return SCBuilder(g)


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
    if not os.path.exists(filename):
        print("Coordinates file not found for the dataset.")
        return None

    df_coords = pd.read_csv(filename, sep=delimeter)
    df_coords.columns = [s.strip() for s in df_coords.columns]

    # create a dictionary of coordinates (node_id : (x, y))
    return dict(
        zip(
            df_coords[node_id_col],
            zip(df_coords[x_col], df_coords[y_col]),
        )
    )
