import numpy as np
import pytest

from sclibrary.edge_flow_denoising import EdgeFlowDenoising
from sclibrary.simplicial_complex import SimplicialComplexNetwork


@pytest.fixture
def f0():
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


@pytest.fixture
def f():
    return np.array(
        [2.90, 0.25, 1.78, -1.50, 1.76, 1.53, 1.32, 0.08, 0.67, 1.73]
    )


class TestEdgeFlowDenoising:

    def test_edge_flow_denoising_P_L1(
        self, sc: SimplicialComplexNetwork, f0: np.ndarray, f: np.ndarray
    ):

        P_choice = "L1"

        edge_flow_denoising = EdgeFlowDenoising(sc)
        edge_flow_denoising.denoise(f0, f, P_choice=P_choice)

        expected_error = 0.70
        assert edge_flow_denoising.errors is not None
        assert np.isclose(
            round(edge_flow_denoising.errors[0], 2),
            expected_error,
        )

        assert edge_flow_denoising.f_estimated is not None
        f_expected = np.array(
            [1.01, 0.21, 0.75, -0.31, 0.73, 0.74, 1.19, 0.12, 0.51, 0.84]
        )
        assert np.allclose(
            np.round(edge_flow_denoising.f_estimated, 2),
            f_expected,
        )

    def test_edge_flow_denoising_P_L1L(
        self, sc: SimplicialComplexNetwork, f0: np.ndarray, f: np.ndarray
    ):

        P_choice = "L1L"

        edge_flow_denoising = EdgeFlowDenoising(sc)
        edge_flow_denoising.denoise(f0, f, P_choice=P_choice)

        expected_error = 0.73
        assert edge_flow_denoising.errors is not None
        assert np.isclose(
            round(edge_flow_denoising.errors[0], 2),
            expected_error,
        )

        assert edge_flow_denoising.f_estimated is not None
        f_expected = np.array(
            [1.26, 0.05, 0.65, -0.06, 0.83, 0.74, 1.19, 0.35, 0.28, 1.07]
        )
        assert np.allclose(
            np.round(edge_flow_denoising.f_estimated, 2),
            f_expected,
        )

    def test_edge_flow_denoising_P_error(
        self, sc: SimplicialComplexNetwork, f0: np.ndarray, f: np.ndarray
    ):

        P_choice = "LL"

        edge_flow_denoising = EdgeFlowDenoising(sc)

        # catch ValueError
        with pytest.raises(ValueError):
            edge_flow_denoising.denoise(f0, f, P_choice=P_choice)
