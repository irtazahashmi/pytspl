import numpy as np
import pytest

from sclibrary.grid_based_filter_design import GridBasedFilterDesign
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

    def test_general_filter(
        self, grid_filter: GridBasedFilterDesign, f: np.ndarray
    ):

        L1, L2 = 1, 1
        f_est_h, f_est_c, f_est_g = grid_filter.general_filter(
            L1=L1, L2=L2, f=f
        )
        f_est = f_est_h + f_est_c + f_est_g

        assert grid_filter.history["L1"] is not None
        assert grid_filter.history["L2"] is not None

        assert np.allclose(
            np.round(f_est, 2),
            f,
        )
