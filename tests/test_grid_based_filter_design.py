import numpy as np
import pytest

from sclibrary import SimplicialComplexNetwork
from sclibrary.filters import GridBasedFilterDesign


@pytest.fixture(autouse=True)
def grid_filter(sc: SimplicialComplexNetwork):
    return GridBasedFilterDesign(sc)


class TestGridBasedFilterDesign:

    def test_power_iteration_algo(self, grid_filter: GridBasedFilterDesign):

        P = grid_filter.sc.hodge_laplacian_matrix(rank=1)
        v = grid_filter._power_iteration(P=P, iterations=50)

        assert v is not None
        assert isinstance(v, np.ndarray)

        expected = np.array(
            [0.06, -0.33, 0.15, -0.39, 0.48, 0.49, -0.31, 0.32, 0.12, -0.2]
        )
        assert np.allclose(np.round(v, 2), expected)

    def test_subcomponent_extraction_L1(
        self, grid_filter: GridBasedFilterDesign, f: np.ndarray, f0: np.ndarray
    ):
        p_choice = "L1"
        filter_size = 4

        grid_filter.subcomponent_extraction(
            p_choice=p_choice, L=filter_size, component="gradient", f=f
        )

        # none of the attributes should be None
        for _, result in grid_filter.history.items():
            assert result is not None

        # decreasing error
        assert np.all(
            np.diff(grid_filter.history["error_per_filter_size"]) < 0.1
        )

        f_estimated = grid_filter.history["f_estimated"]
        error = grid_filter.calculate_error(f_estimated, f0)
        excepted_error = 0.70
        assert np.allclose(error, excepted_error, atol=0.01)

    def test_subcomponent_extraction_L1L(
        self, grid_filter: GridBasedFilterDesign, f: np.ndarray, f0: np.ndarray
    ):
        p_choice = "L1L"
        filter_size = 4

        grid_filter.subcomponent_extraction(
            p_choice=p_choice, L=filter_size, component="gradient", f=f
        )

        # none of the attributes should be None
        for _, result in grid_filter.history.items():
            assert result is not None

        # decreasing error
        assert np.all(
            np.diff(grid_filter.history["error_per_filter_size"]) < 0.1
        )

        f_estimated = grid_filter.history["f_estimated"]
        error = grid_filter.calculate_error(f_estimated, f0)
        excepted_error = 0.73
        assert np.allclose(error, excepted_error, atol=0.01)

    def test_subcomponent_extraction_L1_large_filter_order(
        self, grid_filter: GridBasedFilterDesign, f: np.ndarray, f0: np.ndarray
    ):
        p_choice = "L1"
        filter_size = 12

        grid_filter.subcomponent_extraction(
            p_choice=p_choice, L=filter_size, component="gradient", f=f
        )

        # none of the attributes should be None
        for _, result in grid_filter.history.items():
            assert result is not None

        # decreasing error
        assert np.all(
            np.diff(grid_filter.history["error_per_filter_size"]) < 0.1
        )

        error = grid_filter.history["error_per_filter_size"][-1]
        excepted_error = 0.0014
        assert np.allclose(error, excepted_error, atol=1e-4)

    def test_history_subcomponent_extractione(
        self, grid_filter: GridBasedFilterDesign, f: np.ndarray
    ):
        p_choice = "L1"
        filter_size = 4
        component = "gradient"

        grid_filter.subcomponent_extraction(
            p_choice=p_choice, L=filter_size, component=component, f=f
        )

        assert grid_filter.history["filter"] is not None
        assert isinstance(grid_filter.history["filter"], np.ndarray)

        assert grid_filter.history["f_estimated"] is not None
        assert isinstance(grid_filter.history["f_estimated"], np.ndarray)

        assert grid_filter.history["frequency_responses"] is not None
        assert isinstance(
            grid_filter.history["frequency_responses"], np.ndarray
        )

        assert grid_filter.history["error_per_filter_size"] is not None
        assert isinstance(
            grid_filter.history["error_per_filter_size"], np.ndarray
        )

    def test_general_filter(
        self, grid_filter: GridBasedFilterDesign, f: np.ndarray
    ):

        L1, L2 = 1, 1
        f_est_h, f_est_c, f_est_g = grid_filter.general_filter(
            L1=L1, L2=L2, f=f
        )
        f_est = f_est_h + f_est_c + f_est_g

        assert np.allclose(
            np.round(f_est, 2),
            f,
        )

    def test_history_general_filter(
        self, grid_filter: GridBasedFilterDesign, f: np.ndarray
    ):
        L1, L2 = 1, 1
        grid_filter.general_filter(L1=L1, L2=L2, f=f)

        assert grid_filter.history["L1"] is not None
        assert grid_filter.history["L2"] is not None
