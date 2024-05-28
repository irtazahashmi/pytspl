import numpy as np
import pandas as pd
import pytest

from sclibrary import dataset_loader
from sclibrary.io.network_reader import read_B1_B2, read_B2


@pytest.fixture(scope="module")
def sc_mock():
    """
    Read the paper data and return the mock simplicial complex.

    Yields:
        SimplicialComplex: A simplicial complex network object.
    """
    simplical_complex, _ = dataset_loader.load_paper_data()
    yield simplical_complex


@pytest.fixture(scope="module")
def coordinates_mock():
    """
    Read the coordinates of the paper data and return the coordinates.

    Yields:
        dict: Coordinates of the simplicial complex.
    """
    _, coordinates = dataset_loader.load_paper_data()
    yield coordinates


@pytest.fixture(scope="module")
def f0_mock():
    """
    True flow for the mock simplicial complex.
    Used for testing purposes.

    Returns:
        np.ndarray: True flow.
    """
    yield np.array(
        [
            2.25,
            0.13,
            1.72,
            -2.12,
            1.59,
            1.08,
            -0.30,
            -0.21,
            1.25,
            1.45,
        ]
    )


@pytest.fixture(scope="module")
def f_mock():
    """
    Noisy flow for the mock simplicial complex.
    Used for testing purposes.

    Returns:
        np.ndarray: Noisy flow.
    """
    yield np.array(
        [2.90, 0.25, 1.78, -1.50, 1.76, 1.53, 1.32, 0.08, 0.67, 1.73]
    )


@pytest.fixture(scope="module")
def sc_chicago_mock():
    """
    Read the Chicago data and return the mock simplicial complex.

    Yields:
        SimplicialComplex: A simplicial complex network object.
    """
    B1_filename = "data/test_dataset/B1_chicago_sketch.csv"
    B2_filename = "data/test_dataset/B2t_chicago_sketch.csv"

    scbuilder, triangles = read_B1_B2(
        B1_filename=B1_filename, B2_filename=B2_filename
    )
    sc = scbuilder.to_simplicial_complex(triangles=triangles)
    yield sc


@pytest.fixture(scope="module")
def f0_chicago_mock():
    """
    True flow for the Chicago mock simplicial complex.
    Used for testing purposes.

    Returns:
        np.ndarray: True flow.
    """
    flow_path = "data/test_dataset/flow_chicago_sketch.csv"
    flow = (
        pd.read_csv(flow_path, delimiter=",", header=None).to_numpy().flatten()
    )
    yield flow
