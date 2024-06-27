import numpy as np
import pandas as pd
import pytest

from sclibrary import load_dataset


@pytest.fixture(scope="module")
def sc_mock():
    """
    Read the paper data and return the mock simplicial complex.

    Yields:
        SimplicialComplex: A simplicial complex network object.
    """
    simplical_complex, _, _ = load_dataset("paper")
    yield simplical_complex


@pytest.fixture(scope="module")
def coordinates_mock():
    """
    Read the coordinates of the paper data and return the coordinates.

    Yields:
        dict: Coordinates of the simplicial complex.
    """
    _, coordinates, _ = load_dataset("paper")
    yield coordinates


@pytest.fixture(scope="module")
def f0_mock():
    """
    True flow for the mock simplicial complex.
    Used for testing purposes.

    Returns:
        np.ndarray: True flow.
    """
    _, _, flow = load_dataset("paper")
    flow = np.asarray(list(flow.values()))
    yield flow


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
    sc, _, _ = load_dataset("chicago-sketch")
    yield sc


@pytest.fixture(scope="module")
def f0_chicago_mock():
    """
    True flow for the Chicago mock simplicial complex.
    Used for testing purposes.

    Returns:
        np.ndarray: True flow.
    """
    _, _, flow = load_dataset("chicago-sketch")
    flow = np.asarray(list(flow.values()))
    yield flow
