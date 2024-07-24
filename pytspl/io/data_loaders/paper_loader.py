import pandas as pd
import pkg_resources

from pytspl.io.network_reader import read_coordinates, read_csv

PAPER_DATA_FOLDER = pkg_resources.resource_filename(
    "pytspl", "data/paper_data"
)


def load_paper_data() -> tuple:
    """
    Read the paper data and return the simplicial complex, coordinates
    and the flow.

    Returns:
        tuple:
            SimplicialComplex: The simplicial complex of the paper data.
            dict: The coordinates of the nodes.
            dict: The flow data of the paper data.
    """
    # read network data
    filename = PAPER_DATA_FOLDER + "/edges.csv"
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
    filename = PAPER_DATA_FOLDER + "/coordinates.csv"
    coordinates = read_coordinates(
        filename=filename,
        node_id_col="Id",
        x_col="X",
        y_col="Y",
        delimeter=" ",
    )

    # read flow data
    filename = PAPER_DATA_FOLDER + "/flow.csv"
    flow = pd.read_csv(filename, header=None).values[:, 0]
    flow = {edge: flow[i] for i, edge in enumerate(sc.edges)}

    return sc, coordinates, flow
