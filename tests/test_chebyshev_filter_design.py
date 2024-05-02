import numpy as np
import pytest

from sclibrary import SimplicialComplexNetwork
from sclibrary.filters import ChebyshevFilterDesign


@pytest.fixture(autouse=True)
def chebyshev_filter(sc: SimplicialComplexNetwork):
    return ChebyshevFilterDesign(sc)


class TestChebyshevFilterDesign:

    def test_power_iteration_algo(
        self, chebyshev_filter: ChebyshevFilterDesign
    ):

        P = chebyshev_filter.sc.lower_laplacian_matrix(rank=1)
        v = chebyshev_filter._power_iteration(P=P, iterations=50)

        assert v is not None
        assert isinstance(v, np.ndarray)

        expected = np.array(
            [0.06, -0.33, 0.15, -0.39, 0.48, 0.49, -0.31, 0.32, 0.12, -0.2]
        )
        assert np.allclose(np.round(v, 2), expected)

    def test_logistic_function(self, chebyshev_filter: ChebyshevFilterDesign):
        cut_off_frequency = 0.01
        steep = 100
        logistic_func = chebyshev_filter._logistic_function(
            cut_off_frequency, steep
        )
        # Test for a few sample values
        lam_values = np.array([0.01])
        expected_output = np.array([0.5])
        output = logistic_func(lam_values)

        assert np.allclose(output, expected_output)

    def test_get_chebyshev_series(
        self,
        sc: SimplicialComplexNetwork,
        chebyshev_filter: ChebyshevFilterDesign,
    ):
        n = len(sc.hodge_laplacian_matrix())
        domain_min = 0
        _, domain_max = chebyshev_filter.get_alpha(p_choice="L1L")
        g_cheb = chebyshev_filter._get_chebyshev_series(
            n=n, domain_min=domain_min, domain_max=domain_max
        )
        expected_output = np.array(
            [
                0.95938561,
                0.08122877,
                -0.08122876,
                0.08122875,
                -0.08122874,
                0.08122872,
                -0.08122871,
                0.0812287,
                -0.08122869,
                0.04061435,
            ]
        )
        assert np.allclose(g_cheb.funs[0].coeffs, expected_output)

    def test_get_alpha(self, chebyshev_filter: ChebyshevFilterDesign):
        alpha, lambda_max = chebyshev_filter.get_alpha(p_choice="L1L")
        expected_alpha, expected_lambda_max = 2.74, 5.49
        assert np.round(alpha, 2) == expected_alpha
        assert np.round(lambda_max, 2) == expected_lambda_max

    def test_apply_filter(
        self, chebyshev_filter: ChebyshevFilterDesign, f: np.ndarray
    ):
        k = 10
        component = "gradient"
        p_choice = "L1L"
        chebyshev_filter.apply(
            f=f, component=component, p_choice=p_choice, k_trunc_order=k
        )
        error = chebyshev_filter.history["error_per_filter_size"][-1]
        f_estimated = chebyshev_filter.history["f_estimated"][-1]

        expected_error = 0.07
        expected_f = np.array(
            [
                2.63,
                0.43,
                1.83,
                -1.88,
                1.46,
                1.67,
                1.05,
                -0.39,
                0.79,
                1.48,
            ]
        )
        assert np.round(error, 2) == expected_error
        assert np.allclose(np.round(f_estimated, 2), expected_f)
