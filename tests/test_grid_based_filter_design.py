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
        v = grid_filter._power_iteration(iterations=50)

        assert v is not None
        assert isinstance(v, np.ndarray)

        expected = np.array(
            [0.06, -0.33, 0.15, -0.39, 0.48, 0.49, -0.31, 0.32, 0.12, -0.2]
        )
        assert np.allclose(np.round(v, 2), expected)

    def test_apply_filter(
        self, grid_filter: GridBasedFilterDesign, f0: np.ndarray, f: np.ndarray
    ):

        grid_filter.apply_filter(f0, f=f)

        assert grid_filter.f_estimated is not None
        assert grid_filter.errors is not None
        assert grid_filter.frequency_responses is not None

        # decreasing error
        assert np.all(np.diff(grid_filter.errors) < 0.1)

        f_expected = np.array(
            [1.01, 0.2, 0.75, -0.31, 0.73, 0.74, 1.19, 0.12, 0.5, 0.84]
        )

        assert np.allclose(np.round(grid_filter.f_estimated, 2), f_expected)
