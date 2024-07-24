import numpy as np
import pytest

from pytspl.decomposition.hodgedecomposition import *
from pytspl.io import read_csv
from pytspl.simplicial_complex import SimplicialComplex


@pytest.fixture
def flow():
    yield np.array(
        [0.03, 0.5, 2.38, 0.88, -0.53, -0.52, 1.08, 0.47, -1.17, 0.09]
    )


class TestHodgeDecompostion:

    def test_harmonic_component_condition(
        self, sc_mock: SimplicialComplex, flow: np.ndarray
    ):
        # L1@f_h = 0
        B1 = sc_mock.incidence_matrix(rank=1)
        B2 = sc_mock.incidence_matrix(rank=2)
        f_h = get_harmonic_flow(B1, B2, flow, round_fig=False)
        L1 = B1.T @ B1
        assert np.allclose(L1 @ f_h, 0)

        # L1L @ f_h = 0
        L1L = sc_mock.lower_laplacian_matrix(rank=1)
        assert np.allclose(L1L @ f_h, 0)

        # L1U @ f_h = 0
        L1U = sc_mock.upper_laplacian_matrix(rank=1)
        assert np.allclose(L1U @ f_h, 0)

    def test_harmonic_component_sig_fig(
        self, sc_mock: SimplicialComplex, flow: np.ndarray
    ):
        # L1@f_h = 0
        B1 = sc_mock.incidence_matrix(rank=1)
        B2 = sc_mock.incidence_matrix(rank=2)
        f_h = get_harmonic_flow(B1, B2, flow, round_fig=True)
        L1 = B1.T @ B1
        assert np.allclose(L1 @ f_h, 0, atol=1e-1)

    def test_curl_component(
        self, sc_mock: SimplicialComplex, flow: np.ndarray
    ):
        # L1L@f_c = 0
        B2 = sc_mock.incidence_matrix(rank=2)
        f_c = get_curl_flow(B2, flow, round_fig=False)
        L1L = sc_mock.lower_laplacian_matrix(rank=1)
        assert np.allclose(L1L @ f_c, 0)

    def test_curl_component_sig_fig(
        self, sc_mock: SimplicialComplex, flow: np.ndarray
    ):
        # L1L@f_c = 0
        B2 = sc_mock.incidence_matrix(rank=2)
        f_c = get_curl_flow(B2, flow, round_fig=True)
        L1L = sc_mock.lower_laplacian_matrix(rank=1)
        assert np.allclose(L1L @ f_c, 0, atol=1e-1)

    def test_gradient_component(
        self, sc_mock: SimplicialComplex, flow: np.ndarray
    ):
        # L1U@f_g = 0
        B1 = sc_mock.incidence_matrix(rank=1)
        f_g = get_gradient_flow(B1, flow, round_fig=False)
        L1U = sc_mock.upper_laplacian_matrix(rank=1)
        assert np.allclose(L1U @ f_g, 0)

    def test_gradient_component_sig_fig(
        self, sc_mock: SimplicialComplex, flow: np.ndarray
    ):
        # L1U@f_g = 0
        B1 = sc_mock.incidence_matrix(rank=1)
        f_g = get_gradient_flow(B1, flow, round_fig=True)
        L1U = sc_mock.upper_laplacian_matrix(rank=1)
        assert np.allclose(L1U @ f_g, 0, atol=1e-1)
