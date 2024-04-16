import networkx as nx

from sclibrary import (
    get_coordinates,
    read_csv,
    read_incidence_matrix,
    read_tntp,
)
from sclibrary.simplicial_complex.extended_graph import ExtendedGraph

NODES = 5
EDGES = 7
INCIDENCE_MAT = [
    [1, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 0, 1],
]


class TestNetworkReader:

    def test_read_csv(self):
        filename = "data/sample_data/edges.csv"
        delimeter = " "
        src_col = "Source"
        dest_col = "Target"
        feature_cols = ["Distance"]
        g = read_csv(filename, delimeter, src_col, dest_col, feature_cols)
        assert isinstance(g, ExtendedGraph)
        assert len(g.nodes) == NODES
        assert len(g.edges) == EDGES
        # incidence matrix
        incidence_mat = nx.incidence_matrix(g).toarray()
        assert incidence_mat.shape[0] == NODES
        assert incidence_mat.shape[1] == EDGES
        assert (incidence_mat == INCIDENCE_MAT).all()

    def test_read_tntp(self):
        filename = "data/sample_data/network.tntp"
        g = read_tntp(
            filename=filename,
            delimeter="\t",
            src_col="Tail",
            dest_col="Head",
            skip_rows=5,
        )
        assert isinstance(g, ExtendedGraph)
        assert len(g.nodes) == NODES
        assert len(g.edges) == EDGES
        # incidence matrix
        incidence_mat = nx.incidence_matrix(g).toarray()
        assert incidence_mat.shape[0] == NODES
        assert incidence_mat.shape[1] == EDGES
        assert (incidence_mat == INCIDENCE_MAT).all()

    def test_read_incidence_matrix(self):
        B1_filename = "data/sample_data/B1.csv"
        B2_filename = "data/sample_data/B2.csv"
        g = read_incidence_matrix(B1_filename, B2_filename)
        assert isinstance(g, ExtendedGraph)
        assert len(g.nodes) == NODES
        assert len(g.edges) == EDGES
        # incidence matrix
        incidence_mat = nx.incidence_matrix(g).toarray()
        assert incidence_mat.shape[0] == NODES
        assert incidence_mat.shape[1] == EDGES
        assert (incidence_mat == INCIDENCE_MAT).all()

    def test_get_coordinates(self):
        filename = "data/sample_data/coordinates.csv"
        coordinates = get_coordinates(
            filename=filename,
            node_id_col="Id",
            x_col="X",
            y_col="Y",
            delimeter=" ",
        )
        assert len(coordinates) == NODES
