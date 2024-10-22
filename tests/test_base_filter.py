import numpy as np
import pytest

from pytspl.filters.base_filter import BaseFilter
from pytspl.simplicial_complex import SimplicialComplex


@pytest.fixture(autouse=True)
def filter(sc_mock: SimplicialComplex):
    return BaseFilter(sc_mock)


class TestBaseFilter:

    def test_get_true_signal_gradient(
        self, filter: BaseFilter, f_mock: np.ndarray
    ):
        component = "gradient"
        f_true = filter.get_true_signal(f_mock, component)

        f_expected = np.array(
            [2.48, 0.56, 1.89, -1.92, 1.34, 1.84, 1.01, -0.51, 0.95, 1.45]
        )
        assert np.allclose(
            np.round(f_true, 2),
            f_expected,
        )

    def test_calcualte_error(
        self, filter: BaseFilter, f0_mock: np.ndarray, f_mock: np.ndarray
    ):
        excepted_error = 0.46
        error = filter.calculate_error_NRMSE(f_mock, f0_mock)
        assert np.isclose(error, excepted_error, atol=1e-2)

    def test_power_iteration_algorithm(
        self, filter: BaseFilter, sc_mock: SimplicialComplex
    ):
        P = sc_mock.hodge_laplacian_matrix(rank=1).toarray()
        v = filter.power_iteration(P=P, iterations=50)
        lambda_max = np.mean((P @ v) / v)

        assert isinstance(v, np.ndarray)

        expected = np.array(
            [0.06, -0.33, 0.15, -0.39, 0.48, 0.49, -0.31, 0.32, 0.12, -0.2]
        )
        assert np.allclose(np.round(v, 2), expected)

        expected_lambda_max = 5.48798
        assert np.isclose(lambda_max, expected_lambda_max, atol=1e-6)

    def test_get_component_coefficients(self, filter: BaseFilter):
        alpha_g = filter.get_component_coefficients(component="gradient")
        alpha_c = filter.get_component_coefficients(component="curl")
        alpha_h = filter.get_component_coefficients(component="harmonic")

        expected_g = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 1])
        expected_c = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
        expected_h = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        assert np.array_equal(alpha_g, expected_g)
        assert np.array_equal(alpha_c, expected_c)
        assert np.array_equal(alpha_h, expected_h)

    def test_get_component_coefficients_error(self, filter: BaseFilter):
        component = "unknown"

        with pytest.raises(ValueError):
            filter.get_component_coefficients(component=component)
