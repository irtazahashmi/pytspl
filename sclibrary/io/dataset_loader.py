import os
import pprint

import networkx as nx
import pandas as pd

from sclibrary.io.network_reader import get_coordinates, read_csv, read_tntp

"""Module for loading transportation network datasets."""

DATA_FOLDER = "data/transportation_networks"
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
    features = list(metadeta.iloc[4].values[1:])
    # remove trailing whitespace in the feature names
    features = [feature.strip() for feature in features if feature != ";"]

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
        Returns an empty DataFrame if the flow data file is not found.
    """
    flow_data_path = f"{DATA_FOLDER}/{dataset}/{dataset}_flow.tntp"

    flow = None
    if os.path.exists(f"{DATA_FOLDER}/{dataset}/{dataset}_flow.tntp"):
        flow = pd.read_csv(flow_data_path, sep="\t")
    else:
        print(f"Flow data file not found for the dataset: {dataset}")
        return pd.DataFrame()

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

    pprint.pprint(get_dataset_summary(dataset=dataset))

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
    # generate coordinates using spring layout if coordinates are not provided
    if coordinates is None:
        print("Generating coordinates using spring layout.")
        graph = nx.Graph(sc.edges)
        coordinates = nx.spring_layout(graph)

    # read the flow data
    flow = load_flow(dataset=dataset)
    flow_dict = {}
    if not flow.empty:
        for _, row in flow.iterrows():
            source, target = row["From "], row["To "]
            if (source, target) in sc.edges:
                flow_dict[(source, target)] = row["Volume "].astype(float)

    return sc, coordinates, flow_dict


def load_paper_data() -> tuple:
    """
    Read the paper data and return the simplicial complex and coordinates.

    Returns:
        tuple: The simplicial complex and the coordinates of the nodes.
    """
    data_folder = "data/paper_data"

    # read csv
    filename = data_folder + "/edges.csv"
    delimeter = " "
    src_col = "Source"
    dest_col = "Target"
    feature_cols = ["Distance"]

    G = read_csv(
        filename=filename,
        delimeter=delimeter,
        src_col=src_col,
        dest_col=dest_col,
        feature_cols=feature_cols,
    )
    sc = G.to_simplicial_complex(
        condition="distance", dist_col_name="Distance", dist_threshold=1.5
    )

    # if coordinates exist
    filename = data_folder + "/coordinates.csv"
    coordinates = get_coordinates(
        filename=filename,
        node_id_col="Id",
        x_col="X",
        y_col="Y",
        delimeter=" ",
    )

    return sc, coordinates
