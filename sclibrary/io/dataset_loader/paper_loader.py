from sclibrary.io.network_reader import read_coordinates, read_csv


def load_paper_data() -> tuple:
    """
    Read the paper data and return the simplicial complex and coordinates.

    Returns:
        tuple:
            SimplicialComplex: The simplicial complex of the paper data.
            dict: The coordinates of the nodes.
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
