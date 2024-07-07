"""Module for preprocessing simplicial complex network data.

The network can be read as the following formats:
- TNTP
- CSV
- B1 and B2 incidence matrices

Once the data is read, the SCBuilder object is created to build the
simplicial complex using the nodes, edges and triangles (based on
the user defined condition).

The module also provides functionality to read the coordinates
and flow data.
"""

import os

import numpy as np
import pandas as pd

from pytspl.simplicial_complex.scbuilder import SCBuilder


def _extract_nodes_edges(
    df: pd.DataFrame, src_col: str, dest_col: str, start_index_zero: bool
) -> list:
    """
    Extract nodes and edges from the network dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the edges.
        src_col (str): The name of the column containing the source nodes.
        dest_col (str): The name of the column containing the destination
        start_index_zero (bool): True, if the node ids start from 0. False,
        if the node ids start from 1.

    Returns:
        list: List of nodes and edges.
    """
    nodes = set()
    edges = []

    for _, row in df.iterrows():
        # subtract 1 to make the node ids 0-indexed
        from_node = int(row[src_col])
        to_node = int(row[dest_col])

        if not start_index_zero:
            from_node -= 1
            to_node -= 1

        nodes.add(from_node)
        nodes.add(to_node)

        if (from_node, to_node) in edges or (to_node, from_node) in edges:
            continue

        edges.append((from_node, to_node))

    nodes = list(range(max(nodes) + 1))
    # order edges
    edges.sort()

    return nodes, edges


def read_tntp(
    filename: str,
    src_col: str,
    dest_col: str,
    skip_rows: int,
    delimeter: str = "\t",
    start_index_zero: bool = True,
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
        start_index_zero (bool): True, if the node ids start from 0. False,
        if the node ids start from 1.

    Returns:
        SCBuilder: SC builder object to build the simplicial complex.
    """
    # Read the file
    df = pd.read_csv(filename, skiprows=skip_rows, sep=delimeter)
    # trimmed cols names
    df.columns = [s.strip() for s in df.columns]

    # And drop the silly first andlast columns
    df.drop(["~", ";"], axis=1, inplace=True)

    # get the nodes and edges
    nodes, edges = _extract_nodes_edges(
        df=df,
        src_col=src_col,
        dest_col=dest_col,
        start_index_zero=start_index_zero,
    )

    # extract features if any
    feature_cols = [
        col for col in df.columns if col not in [src_col, dest_col]
    ]

    edge_features = {}
    node_features = {}
    if len(feature_cols) > 0:
        for i, (from_node, to_node) in enumerate(edges):
            edge_features[(from_node, to_node)] = df.iloc[i][
                feature_cols
            ].to_dict()

    return SCBuilder(
        nodes=nodes,
        edges=edges,
        node_features=node_features,
        edge_features=edge_features,
    )


def read_csv(
    filename: str,
    delimeter: str,
    src_col: str,
    dest_col: str,
    feature_cols: list = None,
    start_index_zero: bool = True,
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
        start_index_zero (bool): True, if the node ids start from 0. False,

    Returns:
       SCBuilder: SC builder object to build the simplicial complex.
    """
    df = pd.read_csv(filename, sep=delimeter)

    # get the nodes and edges
    nodes, edges = _extract_nodes_edges(
        df=df,
        src_col=src_col,
        dest_col=dest_col,
        start_index_zero=start_index_zero,
    )

    # add features if any
    edge_features = {}
    node_features = {}

    if len(feature_cols) > 0:
        for i, (from_node, to_node) in enumerate(edges):
            edge_features[(from_node, to_node)] = df.iloc[i][
                feature_cols
            ].to_dict()

    return SCBuilder(
        nodes=nodes,
        edges=edges,
        node_features=node_features,
        edge_features=edge_features,
    )


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
    assert isinstance(edges, np.ndarray), "Edges should be a numpy array."

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


def read_B1_B2(B1_filename: str, B2_filename: str) -> tuple:
    """
    Read the B1 and B2 incidence matrices.

    Args:
        B1_filename (str): The name of the B1 incidence matrix file.
        B2_filename (str): The name of the B2 incidence matrix file.

    Returns:
        SCBuilder: SC builder object to build the simplicial complex.
        list: List of triangles (2-simplices).
    """
    B1 = pd.read_csv(B1_filename, header=None).to_numpy()

    num_edges = B1.shape[1]
    nodes = set()
    edges = []

    for j in range(num_edges):
        col = B1[:, j]
        from_node = np.where(col == -1)[0][0]
        to_node = np.where(col == 1)[0][0]

        nodes.add(from_node)
        nodes.add(to_node)

        edges.append((from_node, to_node))

    nodes = list(range(max(nodes) + 1))
    edges.sort()

    scbuilder = SCBuilder(nodes=nodes, edges=edges)
    triangles = read_B2(B2_filename, np.asarray(edges))

    return scbuilder, triangles


def read_coordinates(
    filename: str,
    node_id_col: str,
    x_col: str,
    y_col: str,
    delimeter: str,
    start_index_zero: bool = True,
) -> dict:
    """
    Read a csv file and returns a dictionary of coordinates.

    Args:
        filename (str): The name of the file.
        node_id_col (str): The name of the column containing the node ids.
        x_col (str): The name of the column containing the x coordinates.
        y_col (str): The name of the column containing the y coordinates.
        delimeter (str, optional): The delimeter used in the csv file.
        start_index_zero (bool): True, if the node ids start from 0. False,
        if the node ids start from 1.

    Returns:
        dict: A dictionary of coordinates (node_id : (x, y)).
    """
    if not os.path.exists(filename):
        return None

    df_coords = pd.read_csv(filename, sep=delimeter)
    df_coords.columns = [s.strip() for s in df_coords.columns]

    if not start_index_zero:
        # subtract 1 to make the node ids 0-indexed
        df_coords[node_id_col] = df_coords[node_id_col] - 1

    # create a dictionary of coordinates (node_id : (x, y))
    return dict(
        zip(
            df_coords[node_id_col],
            zip(df_coords[x_col], df_coords[y_col]),
        )
    )


def read_flow(filename: str) -> dict:
    """
    Read the flow.

    Args:
        filename (str): The name of the flow file.

    Returns:
        pd.DataFrame: The flow data.
    """
    flow = pd.DataFrame()
    if os.path.exists(filename):
        flow = pd.read_csv(filename, sep="\t")
    else:
        print("WARNING: Flow data file not found.")

    return flow
