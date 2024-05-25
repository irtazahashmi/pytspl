import numpy as np
import pytest

from sclibrary import SimplicialComplex
from sclibrary.filters.filter import Filter


@pytest.fixture(autouse=True)
def filter(sc: SimplicialComplex):
    return Filter(sc)


class TestFilter:

    def test_get_true_signal_gradient(self, filter: Filter, f: np.ndarray):
        component = "gradient"
        f_true = filter.get_true_signal(f, component)

        f_expected = np.array(
            [2.48, 0.56, 1.89, -1.92, 1.34, 1.84, 1.01, -0.51, 0.95, 1.45]
        )
        assert np.allclose(
            np.round(f_true, 2),
            f_expected,
        )

    def test_calcualte_error(
        self, filter: Filter, f0: np.ndarray, f: np.ndarray
    ):
        excepted_error = 0.46
        error = filter.calculate_error(f, f0)
        assert np.isclose(error, excepted_error, atol=1e-2)

    def test_power_iteration_algo(
        selfl, filter: Filter, sc: SimplicialComplex
    ):

        P = sc.hodge_laplacian_matrix(rank=1).toarray()
        v = filter.power_iteration(P=P, iterations=50)

        assert v is not None
        assert isinstance(v, np.ndarray)

        expected = np.array(
            [0.06, -0.33, 0.15, -0.39, 0.48, 0.49, -0.31, 0.32, 0.12, -0.2]
        )
        assert np.allclose(np.round(v, 2), expected)
