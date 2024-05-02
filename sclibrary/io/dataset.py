import os

import pandas as pd

from sclibrary.io.network_reader import get_coordinates, read_tntp

"""Module for loading transportation network datasets."""

DATA_FOLDER = "data/transportation_networks"


def get_dataset_summary(dataset: str) -> dict:
    """Get the summary of the dataset.

    Args:
        dataset (str): The name of the dataset.

    Returns:
        dict: The summary of the dataset.
    """

    network_data_path = f"{DATA_FOLDER}/{dataset}/{dataset}_net.tntp"
    metadeta = pd.read_csv(network_data_path, sep="\t", header=None)

    number_of_zones = metadeta.iloc[0][0].split(" ")[-1]
    number_of_nodes = metadeta.iloc[1][0].split(" ")[-1]
    first_thru_node = metadeta.iloc[2][0].split(" ")[-1]
    number_of_links = metadeta.iloc[3][0].split(" ")[-1]
    features = metadeta.iloc[4].values[1:]

    return {
        "number_of_zones": number_of_zones,
        "number_of_nodes": number_of_nodes,
        "first_thru_node": first_thru_node,
        "number_of_links": number_of_links,
        "features": features,
    }


def load_transportation_dataset(dataset: str) -> tuple:
    """
    Load the transportation dataset and return the simplicial complex
    and coordinates.

    Args:
        dataset (str): The name of the dataset.

    Returns:
        tuple: The simplicial complex, the coordinates of the nodes if
        they exist. Else, None.
    """
    network_data_path = f"{DATA_FOLDER}/{dataset}/{dataset}_net.tntp"
    coordinates_data_path = f"{DATA_FOLDER}/{dataset}/{dataset}_node.tntp"

    print(get_dataset_summary(dataset=dataset))

    sc = read_tntp(
        filename=network_data_path,
        src_col="init_node",
        dest_col="term_node",
        skip_rows=8,
        delimeter="\t",
    ).to_simplicial_complex()

    # check if the coordinates file exists
    coordinates = None
    if os.path.exists(coordinates_data_path):
        coordinates = get_coordinates(
            filename=coordinates_data_path,
            node_id_col="node",
            x_col="X",
            y_col="Y",
            delimeter="\t",
        )

    return sc, coordinates
