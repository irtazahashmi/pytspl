import pytest

from sclibrary.network_reader import NetworkReader
from sclibrary.simplicial_complex import SimplicialComplexNetwork


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

    G = NetworkReader.read_csv(
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
