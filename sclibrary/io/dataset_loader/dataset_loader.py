"""Module for preprocessing and loading datasets for analysis.
"""

from .forex_loader import load_forex_data
from .paper_loader import load_paper_data
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

    # paper data
    if dataset == "paper":
        return load_paper_data()
    # forex data
    elif dataset == "forex":
        return load_forex_data()
    else:
        return load_transportation_dataset(dataset=dataset)
