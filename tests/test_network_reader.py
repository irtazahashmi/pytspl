import networkx as nx
import numpy as np
import pandas as pd

from sclibrary.io.network_reader import (
    get_coordinates,
    read_B1,
    read_csv,
    read_tntp,
)
from sclibrary.simplicial_complex.extended_graph import ExtendedGraph

NODES = 7
EDGES = 10
INCIDENCE_MATRIX = np.asarray(
    [
        [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ]
)


class TestNetworkReader:

    def test_read_csv(self):
        filename = "data/paper_data/edges.csv"
        delimeter = " "
        src_col = "Source"
        dest_col = "Target"
        feature_cols = ["Distance"]
        g = read_csv(filename, delimeter, src_col, dest_col, feature_cols)

        assert isinstance(g, ExtendedGraph)
        assert len(g.nodes) == NODES
        assert len(g.edges) == EDGES

        # incidence matrix
        incidence_mat = nx.incidence_matrix(g, oriented=True).toarray()
        assert incidence_mat.shape[0] == NODES
        assert incidence_mat.shape[1] == EDGES
        assert np.array_equal(incidence_mat, INCIDENCE_MATRIX)

    def test_read_tntp(self):
        filename = "data/paper_data/network.tntp"
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

        incidence_mat = nx.incidence_matrix(g, oriented=True).toarray()
        assert incidence_mat.shape[0] == NODES
        assert incidence_mat.shape[1] == EDGES
        assert np.array_equal(incidence_mat, INCIDENCE_MATRIX)

    def test_B1_test_data(self):
        B1_filename = "data/test_dataset/B1.csv"
        B1 = pd.read_csv(B1_filename, header=None).to_numpy()
        g = read_B1(B1_filename)

        assert isinstance(g, ExtendedGraph)
        assert len(g.nodes) == NODES
        assert len(g.edges) == EDGES

        incidence_mat = nx.incidence_matrix(g, oriented=True).toarray()
        assert np.array_equal(incidence_mat, B1)

    def test_B1_chicago_data(self):
        nodes, edges = 546, 1088
        B1_filename = "data/test_dataset/B1_chicago_sketch.csv"
        B1 = pd.read_csv(B1_filename, header=None).to_numpy()
        L1 = B1 @ B1.T

        g = read_B1(B1_filename)

        assert len(g.nodes) == nodes
        assert len(g.edges) == edges

        sc = g.to_simplicial_complex()
        B1_sc = sc.hodge_laplacian_matrix(rank=0)

        assert np.allclose(L1, B1_sc)

    def test_get_coordinates(self):
        nodes = 546
        filename = "data/test_dataset/coordinates_chicago_sketch.csv"
        coordinates = get_coordinates(
            filename=filename,
            node_id_col="Id",
            x_col="X",
            y_col="Y",
            delimeter=",",
        )
        assert len(coordinates) == nodes
