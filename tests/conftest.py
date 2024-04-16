import numpy as np
import pytest

from sclibrary import SimplicialComplexNetwork, read_csv


@pytest.fixture(scope="module")
def sc():
    """
    Read the edges.csv file and create a simplicial complex network for testing.

    Yields:
        SimplicialComplexNetwork: A simplicial complex network object.
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

    simplices = G.simplicies(
        condition="distance", dist_col_name="Distance", dist_threshold=1.5
    )
    yield SimplicialComplexNetwork(simplices=simplices)


@pytest.fixture(scope="module")
def f0():
    """
    True flow for the simplicial complex above.
    Used for testing purposes.

    Returns:
        np.ndarray: True flow.
    """
    return np.array(
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
def f():
    """
    Noisy flow for the simplicial complex above.
    Used for testing purposes.

    Returns:
        np.ndarray: Noisy flow.
    """
    return np.array(
        [2.90, 0.25, 1.78, -1.50, 1.76, 1.53, 1.32, 0.08, 0.67, 1.73]
    )
