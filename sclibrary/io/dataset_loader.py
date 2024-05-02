import os

import pandas as pd

from sclibrary.io.network_reader import get_coordinates, read_tntp

"""Module for loading transportation network datasets."""

DATA_FOLDER = "data/transportation_networks"
METADATA_ROWS = 8


def list_transportation_datasets() -> list:
    """List the available transportation datasets.

    Returns:
        list: The list of available transportation datasets.
    """
    datasets = os.listdir(DATA_FOLDER)
    # remove README.md file
    datasets.remove("README.md")
    return datasets


def get_dataset_summary(dataset: str) -> dict:
    """Get the summary of the dataset.

    Args:
        dataset (str): The name of the dataset.

    Returns:
        dict: The summary of the dataset.
    """

    network_data_path = f"{DATA_FOLDER}/{dataset}/{dataset}_net.tntp"
    coordinates_data_path = f"{DATA_FOLDER}/{dataset}/{dataset}_node.tntp"
    flow_data_path = f"{DATA_FOLDER}/{dataset}/{dataset}_flow.tntp"

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
        "coordinates_exist": os.path.exists(coordinates_data_path),
        "flow_data_exist": os.path.exists(flow_data_path),
    }


def load_flow(dataset: str) -> pd.DataFrame:
    """Read the flow data of the transportation dataset.

    Args:
        dataset (str): The name of the dataset.

    Returns:
        pd.DataFrame: The flow data of the transportation dataset.
    """
    flow_data_path = f"{DATA_FOLDER}/{dataset}/{dataset}_flow.tntp"

    flow = None
    if os.path.exists(f"{DATA_FOLDER}/{dataset}/{dataset}_flow.tntp"):
        flow = pd.read_csv(flow_data_path, sep="\t")
    else:
        print(f"Flow data file not found for the dataset: {dataset}")

    return flow


def load(dataset: str) -> tuple:
    """
    Load the transportation dataset and return the simplicial complex
    and coordinates.

    Args:
        dataset (str): The name of the dataset.

    Returns:
        tuple: The simplicial complex, the coordinates of the nodes if
        they exist, and the flow data if it exists. Else, the coordinates
        and flow data will be None.
    """
    network_data_path = f"{DATA_FOLDER}/{dataset}/{dataset}_net.tntp"
    coordinates_data_path = f"{DATA_FOLDER}/{dataset}/{dataset}_node.tntp"

    print(get_dataset_summary(dataset=dataset))

    # read the network data
    sc = read_tntp(
        filename=network_data_path,
        src_col="init_node",
        dest_col="term_node",
        skip_rows=METADATA_ROWS,
        delimeter="\t",
    ).to_simplicial_complex()

    # read the coordinates data
    coordinates = get_coordinates(
        filename=coordinates_data_path,
        node_id_col="node",
        x_col="X",
        y_col="Y",
        delimeter="\t",
    )

    # read the flow data
    flow = load_flow(dataset=dataset)
    flow_dict = {}
    for _, row in flow.iterrows():
        source, target = row["From "], row["To "]
        if (source, target) in sc.edges:
            flow_dict[(source, target)] = row["Volume "].astype(float)

    return sc, coordinates, flow_dict
