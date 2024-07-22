"""Module for preprocessing and loading datasets for analysis.
"""

from .data_loaders.forex_loader import load_forex_data
from .data_loaders.lastfm_loader import load_lastfm_1k_artist
from .data_loaders.paper_loader import load_paper_data
from .data_loaders.transportation_loader import (
    list_transportation_datasets,
    load_transportation_dataset,
)
from .data_loaders.wsn_loader import load_wsn_data

DATASETS = {
    "paper": load_paper_data,
    "forex": load_forex_data,
    "lastfm-1k-artist": load_lastfm_1k_artist,
    "wsn": load_wsn_data,
}


def list_datasets() -> list:
    """
    List the available datasets.

    Returns:
        list: The list of available datasets.
    """
    other_datasets = list(DATASETS.keys())
    transportation_datasets = list_transportation_datasets()
    return transportation_datasets + other_datasets


def load_dataset(dataset: str) -> tuple:
    """
    Load the dataset and return the simplicial complex
    and coordinates.

    Args:
        dataset (str): The name of the dataset.

    ValueError:
        If the dataset is not found.

    Returns:
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

    if dataset in DATASETS:
        sc, coordinates, flow = DATASETS[dataset]()
    else:
        sc, coordinates, flow = load_transportation_dataset(dataset=dataset)

    assert sc is not None
    assert coordinates is not None
    assert flow is not None

    # each node should have a coordinate
    assert len(sc.nodes) == len(coordinates)

    # print summary of the dataset
    sc.print_summary()
    print(f"Coordinates: {len(coordinates)}")
    print(f"Flow: {len(flow)}")

    return sc, coordinates, flow
