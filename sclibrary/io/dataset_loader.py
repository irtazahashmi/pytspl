"""Module for preprocessing and loading datasets for analysis.
"""

import pandas as pd

from sclibrary.io.network_reader import read_B1_B2, read_coordinates, read_csv

from .transportation_loader import (
    list_transportation_datasets,
    load_transportation_dataset,
)


def list_datasets() -> list:
    """
    List the available datasets.

    Returns:
        list: The list of available datasets.
    """
    transportation_datasets = list_transportation_datasets()
    other_datasets = ["paper", "forex"]
    return transportation_datasets + other_datasets


def load(dataset: str) -> tuple:
    """
    Load the dataset and return the simplicial complex
    and coordinates.

    Args:
        dataset (str): The name of the dataset.

    ValueError:
        If the dataset is not found.

    Returns:
        tuple:
            SimplicialComplex: The simplicial complex of the dataset.
            dict: The coordinates of the nodes. If the coordinates do not
            exist, the coordinates are generated using spring layout.
            dict: The flow data of the dataset. If the flow data does not
            exist, an empty dictionary is returned.
    """
    datasets = list_datasets()
    if dataset not in datasets:
        raise ValueError(
            f"Dataset {dataset} not found. Available datasets: {datasets}"
        )

    if dataset == "paper":
        return load_paper_data()
    elif dataset == "forex":
        return load_forex_data()
    else:
        return load_transportation_dataset(dataset)


def load_paper_data() -> tuple:
    """
    Read the paper data and return the simplicial complex and coordinates.

    Returns:
        tuple:
            SimplicialComplex: The simplicial complex of the paper data.
            dict: The coordinates of the nodes. If the coordinates do not
            exist, the coordinates are generated using spring layout.
    """
    data_folder = "data/paper_data"

    # read network data
    filename = data_folder + "/edges.csv"
    delimeter = " "
    src_col = "Source"
    dest_col = "Target"
    feature_cols = ["Distance"]

    sc = read_csv(
        filename=filename,
        delimeter=delimeter,
        src_col=src_col,
        dest_col=dest_col,
        feature_cols=feature_cols,
    ).to_simplicial_complex()

    # read coordinates data
    filename = data_folder + "/coordinates.csv"
    coordinates = read_coordinates(
        filename=filename,
        node_id_col="Id",
        x_col="X",
        y_col="Y",
        delimeter=" ",
    )

    return sc, coordinates


def load_forex_data() -> tuple:
    """
    Load the forex data and return the simplicial complex and coordinates.

    Returns:
        tuple:
            SimplicialComplex: The simplicial complex of the forex data.
            dict: The coordinates of the nodes. If the coordinates do not
            exist, the coordinates are generated using spring layout.
    """
    data_folder = "data/foreign_exchange"
    B1_filename = f"{data_folder}/B1.csv"
    B2_filename = f"{data_folder}/B2t.csv"
    y_filename = f"{data_folder}/flow_FX_1538755200.csv"

    scbuilder, triangles = read_B1_B2(
        B1_filename=B1_filename, B2_filename=B2_filename
    )
    sc = scbuilder.to_simplicial_complex(triangles=triangles)

    # no coordinates for forex data
    coordinates = None
    flow = pd.read_csv(y_filename, header=None).values[:, 0]

    return sc, coordinates, flow
