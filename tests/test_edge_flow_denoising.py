from unittest.mock import patch

import numpy as np
import pytest

from sclibrary import SimplicialComplex
from sclibrary.filters import EdgeFlowDenoising


@pytest.fixture(autouse=True)
def denoising(sc: SimplicialComplex):
    return EdgeFlowDenoising(sc)


class TestEdgeFlowDenoising:

    def test_edge_flow_denoising_history(
        self, denoising: EdgeFlowDenoising, f0: np.ndarray, f: np.ndarray
    ):
        p_choice = "L1"
        denoising.denoise(
            p_choice=p_choice,
            f=f,
            f_true=f0,
        )

        # none of the attributes should be None for the history
        for _, result in denoising.history.items():
            assert result is not None

    def test_edge_flow_denoising_P_L1(
        self, denoising: EdgeFlowDenoising, f0: np.ndarray, f: np.ndarray
    ):
        p_choice = "L1"
        denoising.denoise(
            p_choice=p_choice,
            f=f,
            f_true=f0,
        )

        expected_error = 0.70
        actual_error = denoising.history["extracted_component_error"][-1]
        assert np.isclose(
            np.round(actual_error, 2),
            expected_error,
        )

        f_expected = np.array(
            [1.01, 0.21, 0.75, -0.31, 0.73, 0.74, 1.19, 0.12, 0.51, 0.84]
        )
        assert np.allclose(
            np.round(denoising.history["f_estimated"], 2),
            f_expected,
        )

    def test_edge_flow_denoising_mu_values(self, denoising: EdgeFlowDenoising):
        f0 = np.array(
            [
                -0.0689,
                -0.1378,
                0.2067,
                -0.0689,
                0.3445,
                -0.5512,
                0.5512,
                0.3675,
                0.1837,
                -0.1837,
            ]
        )

        f = np.array(
            [
                0.4688,
                1.6961,
                -2.0522,
                0.7933,
                0.6633,
                -1.8589,
                0.1176,
                0.7101,
                3.7621,
                2.5857,
            ]
        )

        mu_values = [
            1e-2,
            5e-2,
            1e-1,
            2.5e-1,
            5e-1,
            1,
            2.5,
            5,
            10,
            25,
            50,
            100,
        ]

        denoising.denoise(f=f, f_true=f0, p_choice="L1", mu_vals=mu_values)
        expected_errors = [
            5.48,
            4.86,
            4.26,
            3.15,
            2.22,
            1.40,
            0.68,
            0.37,
            0.20,
            0.09,
            0.06,
            0.05,
        ]

        assert np.allclose(
            np.round(denoising.history["extracted_component_error"], 2),
            expected_errors,
        )

    def test_edge_flow_denoising_P_L1L(
        self, denoising: EdgeFlowDenoising, f0: np.ndarray, f: np.ndarray
    ):
        p_choice = "L1L"
        denoising.denoise(
            p_choice=p_choice,
            f=f,
            f_true=f0,
        )

        expected_error = 0.73
        actual_error = denoising.history["extracted_component_error"][-1]
        assert np.isclose(
            np.round(actual_error, 2),
            expected_error,
        )

        f_expected = np.array(
            [1.26, 0.05, 0.65, -0.06, 0.83, 0.74, 1.19, 0.35, 0.28, 1.07]
        )
        assert np.allclose(
            np.round(denoising.history["f_estimated"], 2),
            f_expected,
        )

    def test_edge_flow_denoising_P_not_found(
        self, denoising: EdgeFlowDenoising, f0: np.ndarray, f: np.ndarray
    ):
        p_choice = "unknown"
        # catch ValueError
        with pytest.raises(ValueError):
            denoising.denoise(
                p_choice=p_choice,
                f=f,
                f_true=f0,
            )

    def test_plot_desired_frequency_response(
        self, denoising: EdgeFlowDenoising, f0: np.ndarray, f: np.ndarray
    ):
        p_choice = "L1"
        denoising.denoise(
            p_choice=p_choice,
            f=f,
            f_true=f0,
        )

        with patch("matplotlib.pyplot") as mock_plt:
            denoising.plot_desired_frequency_response(p_choice)
            mock_plt.figure.assert_called_once_with(figsize=(10, 6))
            mock_plt.xlabel.assert_called_once_with("Eigenvalues")
            mock_plt.ylabel.assert_called_once_with("Frequency Response")
            mock_plt.title.assert_called_once_with(
                "Desired Frequency Response"
            )

    def test_plot_desired_frequency_response_no_history(
        self, denoising: EdgeFlowDenoising
    ):
        p_choice = "L1"
        # catch ValueError
        with pytest.raises(ValueError):
            denoising.plot_desired_frequency_response(p_choice)
