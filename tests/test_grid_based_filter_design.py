import numpy as np
import pytest

from pytspl.filters import GridBasedFilterDesign
from pytspl.simplicial_complex import SimplicialComplex


@pytest.fixture(autouse=True)
def grid_filter(sc_mock: SimplicialComplex):
    return GridBasedFilterDesign(sc_mock)


class TestGridBasedFilterDesign:

    def test_denoising_L1(
        self,
        grid_filter: GridBasedFilterDesign,
        f_mock: np.ndarray,
        f0_mock: np.ndarray,
    ):
        filter_size = 4
        p_choice = "L1"
        mu = 0.5

        grid_filter.denoise(
            f=f_mock, f_true=f0_mock, p_choice=p_choice, L=filter_size, mu=mu
        )

        error = grid_filter.history["extracted_component_error"][-1]
        excepted_error = 0.70
        assert np.allclose(error, excepted_error, atol=0.01)

    def test_denoising_L1L(
        self,
        grid_filter: GridBasedFilterDesign,
        f_mock: np.ndarray,
        f0_mock: np.ndarray,
    ):
        filter_size = 4
        p_choice = "L1L"
        mu = 0.5

        grid_filter.denoise(
            f=f_mock, f_true=f0_mock, p_choice=p_choice, L=filter_size, mu=mu
        )

        excepted_error = 0.73
        error = grid_filter.history["extracted_component_error"][-1]
        assert np.allclose(error, excepted_error, atol=0.01)

    def test_denoising_L1_large_filter_order(
        self,
        grid_filter: GridBasedFilterDesign,
        f_mock: np.ndarray,
        f0_mock: np.ndarray,
    ):
        p_choice = "L1"
        filter_size = 12

        grid_filter.denoise(
            f=f_mock,
            f_true=f0_mock,
            p_choice=p_choice,
            L=filter_size,
        )

        excepted_error = 0.7026
        error = grid_filter.history["extracted_component_error"][-1]
        assert np.allclose(error, excepted_error, atol=1e-4)

    def test_subcomponent_extraction_L1L_f_est(
        self,
        grid_filter: GridBasedFilterDesign,
        f_mock: np.ndarray,
    ):

        p_choice = "L1L"
        component = "gradient"
        filter_size = 1

        grid_filter.subcomponent_extraction(
            f=f_mock,
            component=component,
            p_choice=p_choice,
            L=filter_size,
        )

        f_est = grid_filter.history["f_estimated"]
        excepted_f_est = np.array(
            [2.69, 0.23, 1.65, -1.39, 1.63, 1.42, 1.22, 0.07, 0.62, 1.6]
        )
        assert np.allclose(np.round(f_est, 2), excepted_f_est)

    def test_subcomponent_extraction_L1L_error(
        self,
        grid_filter: GridBasedFilterDesign,
        f_mock: np.ndarray,
    ):
        p_choice = "L1L"
        component = "gradient"
        filter_size = 1

        grid_filter.subcomponent_extraction(
            f=f_mock,
            component=component,
            p_choice=p_choice,
            L=filter_size,
        )

        excepted_error = 0.23
        error = grid_filter.history["extracted_component_error"][-1]
        assert np.allclose(error, excepted_error, atol=1e-2)

    def test_history_subcomponent_extraction(
        self,
        grid_filter: GridBasedFilterDesign,
        f_mock: np.ndarray,
        f0_mock: np.ndarray,
    ):
        p_choice = "L1"
        component = "gradient"
        filter_size = 4

        grid_filter.subcomponent_extraction(
            f=f_mock,
            component=component,
            p_choice=p_choice,
            L=filter_size,
        )

        assert grid_filter.history["filter"] is not None
        assert isinstance(grid_filter.history["filter"], np.ndarray)

        assert grid_filter.history["f_estimated"] is not None
        assert isinstance(grid_filter.history["f_estimated"], np.ndarray)

        assert grid_filter.history["frequency_responses"] is not None
        assert isinstance(
            grid_filter.history["frequency_responses"], np.ndarray
        )

        assert grid_filter.history["extracted_component_error"] is not None
        assert isinstance(
            grid_filter.history["extracted_component_error"], np.ndarray
        )

    def test_general_filter(
        self,
        grid_filter: GridBasedFilterDesign,
        f_mock: np.ndarray,
        f0_mock: np.ndarray,
    ):

        L1, L2 = 1, 1
        f_est_h, f_est_c, f_est_g = grid_filter.general_filter(
            f=f_mock,
            L1=L1,
            L2=L2,
        )
        f_est = f_est_h + f_est_c + f_est_g

        assert np.allclose(
            np.round(f_est, 2),
            f_mock,
        )
