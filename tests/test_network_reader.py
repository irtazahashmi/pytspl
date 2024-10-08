import numpy as np
import pandas as pd

from pytspl.io.network_reader import (
    read_B1_B2,
    read_coordinates,
    read_csv,
    read_flow,
    read_tntp,
)
from pytspl.simplicial_complex.scbuilder import SCBuilder

PAPER_DATA_FOLDER = "pytspl/data/paper_data"
TRANS_DATA_FOLDER = "pytspl/data/transportation_networks"


NODES = 7
EDGES = 10

INC_MAT_B1 = np.asarray(
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

# from above
INC_MAT_B2 = np.asarray(
    [
        [1, 0, 0],
        [-1, 1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, -1],
        [0, 0, 1],
    ]
)


class TestNetworkReader:

    def test_read_tntp(self):
        filename = f"{PAPER_DATA_FOLDER}/network.tntp"
        scbuilder = read_tntp(
            filename=filename,
            delimiter="\t",
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
        assert np.array_equal(incidence_mat, INC_MAT_B1)

    def test_read_csv(self):
        filename = f"{PAPER_DATA_FOLDER}/edges.csv"
        delimiter = " "
        src_col = "Source"
        dest_col = "Target"
        feature_cols = ["Distance"]
        sc = read_csv(
            filename, delimiter, src_col, dest_col, feature_cols
        ).to_simplicial_complex()

        incidence_mat = sc.incidence_matrix(rank=1).toarray()
        assert incidence_mat.shape[0] == NODES
        assert incidence_mat.shape[1] == EDGES
        assert np.array_equal(incidence_mat, INC_MAT_B1)

    def test_B1_B2_test_data(self):
        B1_filename = f"{PAPER_DATA_FOLDER}/B1.csv"
        B2_filename = f"{PAPER_DATA_FOLDER}/B2t.csv"
        B1 = pd.read_csv(B1_filename, header=None).to_numpy()

        scbuilder, triangles = read_B1_B2(
            B1_filename=B1_filename, B2_filename=B2_filename
        )
        sc = scbuilder.to_simplicial_complex(triangles=triangles)

        assert len(sc.nodes) == NODES
        assert len(sc.edges) == EDGES

        inc_mat_B1 = sc.incidence_matrix(rank=1).toarray()
        assert np.array_equal(inc_mat_B1, B1)

        inc_mat_B2 = sc.incidence_matrix(rank=2).toarray()
        assert np.array_equal(inc_mat_B2, INC_MAT_B2)

    def test_B1_B2_chicago_data(self):
        data_folder = f"{TRANS_DATA_FOLDER}/chicago-sketch"

        nodes, edges = 546, 1088
        B1_filename = f"{data_folder}/B1_chicago_sketch.csv"
        B2_filename = f"{data_folder}/B2t_chicago_sketch.csv"

        B1 = pd.read_csv(B1_filename, header=None).to_numpy()
        B2 = pd.read_csv(B2_filename, header=None).to_numpy().T

        scbuilder, triangles = read_B1_B2(
            B1_filename=B1_filename, B2_filename=B2_filename
        )

        assert isinstance(scbuilder, SCBuilder)
        assert len(scbuilder.nodes) == nodes
        assert len(scbuilder.edges) == edges

        sc = scbuilder.to_simplicial_complex(triangles=triangles)

        B1_calculated = sc.incidence_matrix(rank=1).toarray()
        B2_calculated = sc.incidence_matrix(rank=2).toarray()

        assert np.array_equal(B1, B1_calculated)
        assert np.array_equal(B2, B2_calculated)

    def test_read_coordinates(self):
        data_folder = f"{TRANS_DATA_FOLDER}/chicago-sketch"

        nodes = 546
        filename = f"{data_folder}/coordinates_chicago_sketch.csv"
        coordinates = read_coordinates(
            filename=filename,
            node_id_col="Id",
            x_col="X",
            y_col="Y",
            delimiter=",",
        )
        assert len(coordinates) == nodes

    def test_read_flow_dataset(self):
        data_folder = f"{TRANS_DATA_FOLDER}/siouxfalls"

        dataset = f"{data_folder}/siouxfalls_flow.tntp"
        flow = read_flow(filename=dataset)

        assert isinstance(flow, pd.DataFrame)
        assert flow.empty is False

    def test_read_flow_dataset_not_found(self):
        dataset = "unknown_dataset"
        flow = read_flow(filename=dataset)
        assert flow.empty is True
