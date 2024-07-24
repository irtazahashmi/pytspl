"""Module for preprocessing and loading transportation network
datasets for analysis.
"""

import os

import pandas as pd
import pkg_resources

from pytspl.io.network_reader import (
    read_B1_B2,
    read_coordinates,
    read_flow,
    read_tntp,
)

DATA_FOLDER = pkg_resources.resource_filename(
    "pytspl", "data/transportation_networks"
)

CHICAGO_SKETCH_DATA_FOLDER = pkg_resources.resource_filename(
    "pytspl", "data/transportation_networks/chicago-sketch"
)


METADATA_ROWS = 8


def list_transportation_datasets() -> list:
    """List the available transportation datasets.

    Returns:
        list: The list of available transportation datasets.
    """
    datasets = os.listdir(DATA_FOLDER)
    # remove files
    files = [".DS_Store", "README.md"]
    datasets = [dataset for dataset in datasets if dataset not in files]
    return datasets


def load_flow_transportation(dataset: str, edges: list) -> pd.DataFrame:
    """Read the flow data of the transportation dataset.

    Args:
        dataset (str): The name of the dataset.
        edges (list): The list of edges in the simplicial complex.

    Returns:
        pd.DataFrame: The flow data of the transportation dataset.
        Returns an empty dictionary if the flow data is not found.
    """
    flow_data_path = f"{DATA_FOLDER}/{dataset}/{dataset}_flow.tntp"
    df_flow = read_flow(filename=flow_data_path)
    if df_flow.empty:
        return {}

    visited_nodes = set()
    flow_dict = {}
    if not df_flow.empty:
        for edge in edges:
            source, target = edge
            # index starts at 1
            source += 1
            target += 1

            if (source, target) not in visited_nodes:
                # get the flow volume in the positive direction
                flow_pos = df_flow[
                    (df_flow["From "] == source) & (df_flow["To "] == target)
                ]["Volume "].values[0]

                # check if the flow is in the opposite direction
                try:
                    flow_neg = df_flow[
                        (df_flow["From "] == target)
                        & (df_flow["To "] == source)
                    ]["Volume "].values[0]
                except IndexError:
                    flow_neg = 0

                # calculate the net flow
                net_flow = flow_pos - flow_neg

                # zero index the nodes
                source -= 1
                target -= 1

                flow_dict[(source, target)] = net_flow
                visited_nodes.add((source, target))

    return flow_dict


def load_transportation_dataset(dataset: str) -> tuple:
    """
    Load the transportation dataset and return the simplicial complex
    and coordinates.

    Args:
        dataset (str): The name of the dataset.

    Returns:
        tuple:
            SimplicialComplex: The simplicial complex of the dataset.
            dict: The coordinates of the nodes. If the coordinates do not
            exist, the coordinates are generated using spring layout.
            dict: The flow data of the dataset. If the flow data does not
            exist, an empty dictionary is returned.
    """
    if dataset == "chicago-sketch":
        return load_chicago_sketch()

    start_index_zero = False

    network_data_path = f"{DATA_FOLDER}/{dataset}/{dataset}_net.tntp"
    coordinates_data_path = f"{DATA_FOLDER}/{dataset}/{dataset}_node.tntp"

    # read the network data
    sc = read_tntp(
        filename=network_data_path,
        src_col="init_node",
        dest_col="term_node",
        skip_rows=METADATA_ROWS,
        delimeter="\t",
        # index starts at 1
        start_index_zero=start_index_zero,
    ).to_simplicial_complex()

    # read the coordinates data
    coordinates = read_coordinates(
        filename=coordinates_data_path,
        node_id_col="node",
        x_col="X",
        y_col="Y",
        delimeter="\t",
        start_index_zero=start_index_zero,
    )

    # generate coordinates using spring layout if coordinates are not provided
    if coordinates is None:
        coordinates = sc.generate_coordinates()

    # read the flow data
    flow_dict = load_flow_transportation(dataset=dataset, edges=sc.edges)

    return sc, coordinates, flow_dict


def load_chicago_sketch() -> tuple:
    """
    Load the Chicago sketch dataset straight from the files.

    Returns:
        tuple:
            SimplicialComplex: The simplicial complex of the dataset.
            dict: The coordinates of the nodes.
            dict: The flow data of the dataset.
    """
    B1_dataset_path = f"{CHICAGO_SKETCH_DATA_FOLDER}/B1_chicago_sketch.csv"
    B2_dataset_path = f"{CHICAGO_SKETCH_DATA_FOLDER}/B2t_chicago_sketch.csv"

    scbuilder, triangles = read_B1_B2(B1_dataset_path, B2_dataset_path)
    sc = scbuilder.to_simplicial_complex(triangles=triangles)

    # read coordinates
    coordinates_path = (
        f"{CHICAGO_SKETCH_DATA_FOLDER}/coordinates_chicago_sketch.csv"
    )
    coordinates = read_coordinates(
        coordinates_path,
        node_id_col="Id",
        x_col="X",
        y_col="Y",
        delimeter=",",
        start_index_zero=True,
    )

    # read flow
    flow_path = f"{CHICAGO_SKETCH_DATA_FOLDER}/flow_chicago_sketch.csv"
    flow = (
        pd.read_csv(flow_path, delimiter=",", header=None).to_numpy().flatten()
    )
    # convert to dictionary
    flow_dict = {
        (edge[0], edge[1]): flow[i] for i, edge in enumerate(sc.edges)
    }
    return sc, coordinates, flow_dict
