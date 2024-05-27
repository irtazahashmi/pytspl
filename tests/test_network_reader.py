import numpy as np
import pandas as pd

from sclibrary.io.network_reader import (
    read_B1,
    read_B2,
    read_coordinates,
    read_csv,
    read_flow,
    read_tntp,
)
from sclibrary.simplicial_complex.scbuilder import SCBuilder

NODES = 7
EDGES = 10

INCIDENCE_MATRIX = np.asarray(
    [
        [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ]
)


class TestNetworkReader:

    def test_read_tntp(self):
        filename = "data/paper_data/network.tntp"
        scbuilder = read_tntp(
            filename=filename,
            delimeter="\t",
            src_col="Tail",
            dest_col="Head",
            skip_rows=5,
        )
        assert isinstance(scbuilder, SCBuilder)
        assert len(scbuilder.nodes) == NODES
        assert len(scbuilder.edges) == EDGES

        sc = scbuilder.to_simplicial_complex()

        incidence_mat = sc.incidence_matrix(rank=1).toarray()
        assert incidence_mat.shape[0] == NODES
        assert incidence_mat.shape[1] == EDGES
        assert np.array_equal(incidence_mat, INCIDENCE_MATRIX)

    def test_read_csv(self):
        filename = "data/paper_data/edges.csv"
        delimeter = " "
        src_col = "Source"
        dest_col = "Target"
        feature_cols = ["Distance"]
        sc = read_csv(
            filename, delimeter, src_col, dest_col, feature_cols
        ).to_simplicial_complex()

        incidence_mat = sc.incidence_matrix(rank=1).toarray()
        assert incidence_mat.shape[0] == NODES
        assert incidence_mat.shape[1] == EDGES
        assert np.array_equal(incidence_mat, INCIDENCE_MATRIX)

    def test_B1_test_data(self):
        B1_filename = "data/paper_data/B1.csv"
        B1 = pd.read_csv(B1_filename, header=None).to_numpy()
        sc = read_B1(B1_filename=B1_filename).to_simplicial_complex()

        assert len(sc.nodes) == NODES
        assert len(sc.edges) == EDGES

        incidence_mat = sc.incidence_matrix(rank=1).toarray()
        assert np.array_equal(incidence_mat, B1)

    def test_B1_chicago_data(self):
        nodes, edges = 546, 1088
        B1_filename = "data/test_dataset/B1_chicago_sketch.csv"
        B1 = pd.read_csv(B1_filename, header=None).to_numpy()

        scbuilder = read_B1(B1_filename=B1_filename)
        assert isinstance(scbuilder, SCBuilder)
        assert len(scbuilder.nodes) == nodes
        assert len(scbuilder.edges) == edges

        sc = scbuilder.to_simplicial_complex()
        B1_calculated = sc.incidence_matrix(rank=1).toarray()

        assert np.array_equal(B1, B1_calculated)

    def test_B2_chicago_data(self):
        B1_filename = "data/test_dataset/B1_chicago_sketch.csv"
        B2_filename = "data/test_dataset/B2t_chicago_sketch.csv"
        B2 = pd.read_csv(B2_filename, header=None).to_numpy().T

        scbuilder = read_B1(B1_filename=B1_filename)
        edges = np.asarray(scbuilder.edges)
        triangles = read_B2(B2_filename=B2_filename, edges=edges)
        sc = scbuilder.to_simplicial_complex(triangles=triangles)

        B2_calculated = sc.incidence_matrix(rank=2).toarray()
        assert np.array_equal(B2, B2_calculated)

    def test_read_coordinates(self):
        nodes = 546
        filename = "data/test_dataset/coordinates_chicago_sketch.csv"
        coordinates = read_coordinates(
            filename=filename,
            node_id_col="Id",
            x_col="X",
            y_col="Y",
            delimeter=",",
        )
        assert len(coordinates) == nodes

    def test_read_flow_dataset(self):
        dataset = (
            "data/transportation_networks/siouxfalls/siouxfalls_flow.tntp"
        )
        flow = read_flow(filename=dataset)

        assert isinstance(flow, pd.DataFrame)
        assert flow.empty is False

    def test_read_flow_dataset_not_found(self):
        dataset = "unknown_dataset"
        flow = read_flow(filename=dataset)
        assert flow.empty is True
