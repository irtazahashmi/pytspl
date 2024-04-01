import numpy as np
import pytest

from sclibrary.filters.filter import Filter
from sclibrary.sc.simplicial_complex import SimplicialComplexNetwork


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
def filter(sc: SimplicialComplexNetwork):
    return Filter(sc)


class TestFilter:

    def test_get_true_signal_gradient(self, filter: Filter, f: np.ndarray):
        component = "gradient"
        f_true = filter.get_true_signal(component, f)

        f_expected = np.array(
            [2.48, 0.56, 1.89, -1.92, 1.34, 1.84, 1.01, -0.51, 0.95, 1.45]
        )
        assert np.allclose(
            np.round(f_true, 2),
            f_expected,
        )

    def test_get_true_signal_error(self, filter: Filter, f: np.ndarray):
        component = "unknown"

        with pytest.raises(ValueError):
            filter.get_true_signal(component, f)

    def test_calcualte_error(
        self, filter: Filter, f0: np.ndarray, f: np.ndarray
    ):
        excepted_error = 0.46
        error = filter.calculate_error(f, f0)
        assert np.isclose(error, excepted_error, atol=1e-2)
