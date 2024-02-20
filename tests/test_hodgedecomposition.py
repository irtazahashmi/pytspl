import numpy as np
import pytest

from sclibrary.hodgedecomposition import *
from sclibrary.network_reader import NetworkReader
from sclibrary.simplicial_complex import SimplicialComplexNetwork


@pytest.fixture
def sc():
    data_folder = "data/paper_data"
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


@pytest.fixture
def flow():
    yield [0.03, 0.5, 2.38, 0.88, -0.53, -0.52, 1.08, 0.47, -1.17, 0.09]


class TestHodgeDecompostion:

    def test_harmonic_component_condition(self, sc, flow):
        # L1@f_h = 0
        B1 = sc.incidence_matrix(rank=1)
        B2 = sc.incidence_matrix(rank=2)
        f_h = get_harmonic_flow(B1, B2, flow, round_fig=False)
        L1 = B1.T @ B1
        assert np.allclose(L1 @ f_h, 0)

        # L1L @ f_h = 0
        L1L = sc.lower_laplacian_matrix(rank=1)
        assert np.allclose(L1L @ f_h, 0)

        # L1U @ f_h = 0
        L1U = sc.upper_laplacian_matrix(rank=1)
        assert np.allclose(L1U @ f_h, 0)

    def test_curl_component(self, sc, flow):
        # L1L@f_c = 0
        B2 = sc.incidence_matrix(rank=2)
        f_c = get_curl_flow(B2, flow, round_fig=False)
        L1L = sc.lower_laplacian_matrix(rank=1)
        assert np.allclose(L1L @ f_c, 0)

    def test_gradient_component(self, sc, flow):
        # L1U@f_g = 0
        B1 = sc.incidence_matrix(rank=1)
        f_g = get_gradient_flow(B1, flow, round_fig=False)
        L1U = sc.upper_laplacian_matrix(rank=1)
        assert np.allclose(L1U @ f_g, 0)
