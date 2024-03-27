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
        edf = EdgeFlowDenoising(sc)
        edf.denoise(f0, f, P_choice=P_choice)

        assert edf.history["f_estimated"] is not None
        assert edf.history["error"] is not None
        assert edf.history["frequency_responses"] is not None
        assert edf.history["error_per_filter_size"] is not None

        expected_error = 0.70
        assert np.isclose(
            np.round(edf.history["error"], 2),
            expected_error,
        )

        f_expected = np.array(
            [1.01, 0.21, 0.75, -0.31, 0.73, 0.74, 1.19, 0.12, 0.51, 0.84]
        )
        assert np.allclose(
            np.round(edf.history["f_estimated"], 2),
            f_expected,
        )

    def test_edge_flow_denoising_P_L1L(
        self, sc: SimplicialComplexNetwork, f0: np.ndarray, f: np.ndarray
    ):

        P_choice = "L1L"
        edf = EdgeFlowDenoising(sc)
        edf.denoise(f0, f, P_choice=P_choice)

        assert edf.history["f_estimated"] is not None
        assert edf.history["error"] is not None
        assert edf.history["frequency_responses"] is not None
        assert edf.history["error_per_filter_size"] is not None

        expected_error = 0.73
        assert np.isclose(
            np.round(edf.history["error"], 2),
            expected_error,
        )

        f_expected = np.array(
            [1.26, 0.05, 0.65, -0.06, 0.83, 0.74, 1.19, 0.35, 0.28, 1.07]
        )
        assert np.allclose(
            np.round(edf.history["f_estimated"], 2),
            f_expected,
        )

    def test_edge_flow_denoising_P_error(
        self, sc: SimplicialComplexNetwork, f0: np.ndarray, f: np.ndarray
    ):

        P_choice = "LL"
        edf = EdgeFlowDenoising(sc)

        # catch ValueError
        with pytest.raises(ValueError):
            edf.denoise(f0, f, P_choice=P_choice)
