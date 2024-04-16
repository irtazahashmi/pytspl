import numpy as np
import pytest

from sclibrary import SimplicialComplexNetwork
from sclibrary.filters import EdgeFlowDenoising


class TestEdgeFlowDenoising:

    def test_edge_flow_denoising_P_L1(
        self, sc: SimplicialComplexNetwork, f0: np.ndarray, f: np.ndarray
    ):

        P_choice = "L1"
        edf = EdgeFlowDenoising(sc)
        edf.denoise(
            p_choice=P_choice,
            component="gradient",
            f=f,
        )

        # none of the attributes should be None for the history
        for _, result in edf.history.items():
            assert result is not None

        expected_error = 0.70
        actual_error = edf.calculate_error(edf.history["f_estimated"], f0)
        assert np.isclose(
            np.round(actual_error, 2),
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
        edf.denoise(
            p_choice=P_choice,
            component="gradient",
            f=f,
        )

        # none of the attributes should be None for the history
        for _, result in edf.history.items():
            assert result is not None

        expected_error = 0.73
        actual_error = edf.calculate_error(edf.history["f_estimated"], f0)
        assert np.isclose(
            np.round(actual_error, 2),
            expected_error,
        )

        f_expected = np.array(
            [1.26, 0.05, 0.65, -0.06, 0.83, 0.74, 1.19, 0.35, 0.28, 1.07]
        )
        assert np.allclose(
            np.round(edf.history["f_estimated"], 2),
            f_expected,
        )

    def test_edge_flow_denoising_P_not_found(
        self, sc: SimplicialComplexNetwork, f0: np.ndarray, f: np.ndarray
    ):

        P_choice = "LL"
        edf = EdgeFlowDenoising(sc)

        # catch ValueError
        with pytest.raises(ValueError):
            edf.denoise(
                p_choice=P_choice,
                component="gradient",
                f=f,
            )
