import numpy as np
import pytest

from sclibrary import SimplicialComplex
from sclibrary.filters import ChebyshevFilterDesign


@pytest.fixture(autouse=True)
def chebyshev_filter(sc: SimplicialComplex):
    return ChebyshevFilterDesign(sc)


class TestChebyshevFilterDesign:

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
        sc: SimplicialComplex,
        chebyshev_filter: ChebyshevFilterDesign,
    ):
        n = 10
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

    def test_chebyshev_filter_approximate(
        self, chebyshev_filter: ChebyshevFilterDesign
    ):
        L1L = chebyshev_filter.sc.lower_laplacian_matrix(rank=1).toarray()
        coeffs = np.array(
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
        alpha_g = 2.74
        result = chebyshev_filter._chebyshev_filter_approximate(
            P=L1L, coefficients=coeffs, alpha=alpha_g, k_trnc=1
        )
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, np.eye(len(L1L)) * 0.9593, atol=1e-4)

    def test_get_chebyshev_frequency_approx(
        self, chebyshev_filter: ChebyshevFilterDesign
    ):
        coeffs = np.array(
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
        alpha_g = 2.74
        k = 2
        g_cheb = chebyshev_filter.get_chebyshev_frequency_approx(
            p_choice="L1L", coeffs=coeffs, alpha=alpha_g, k_trunc_order=k
        )
        assert g_cheb is not None
        assert isinstance(g_cheb, np.ndarray)

        expected_first = np.array(
            [0.0, 0.95938561, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        excepted_second = np.array(
            [
                0.02960241,
                0.93736168,
                0.02960241,
                0.02960241,
                -0.02960241,
                -0.02960241,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        assert np.allclose(g_cheb[:, :, 1][0], expected_first, atol=1e-8)
        assert np.allclose(g_cheb[:, :, 1][1], excepted_second, atol=1e-3)

    def test_get_alpha(self, chebyshev_filter: ChebyshevFilterDesign):
        alpha, lambda_max = chebyshev_filter.get_alpha(p_choice="L1L")
        expected_alpha, expected_lambda_max = 2.74, 5.49
        assert np.round(alpha, 2) == expected_alpha
        assert np.round(lambda_max, 2) == expected_lambda_max

    def test_apply_filter(
        self,
        chebyshev_filter: ChebyshevFilterDesign,
    ):
        f = np.array(
            [
                0.0323,
                0.4980,
                2.3825,
                0.8799,
                -0.5297,
                -0.5192,
                1.0754,
                0.4732,
                -1.1667,
                0.0922,
            ]
        )

        k, n = 10, 10
        component = "gradient"
        p_choice = "L1L"
        chebyshev_filter.apply(
            f=f,
            component=component,
            p_choice=p_choice,
            L=k,
        )
        error = chebyshev_filter.history["extracted_component_error"][-1]
        f_estimated = chebyshev_filter.history["f_estimated"][-1]

        expected_error = 0.222
        expected_f = np.array(
            [
                0.2112,
                1.1000,
                1.5808,
                1.0002,
                -0.1685,
                -0.1503,
                0.6551,
                -0.2233,
                -0.9144,
                -0.2253,
            ]
        )

        assert np.allclose(error, expected_error, atol=1e-3)
        assert np.allclose(f_estimated, expected_f, atol=1e-4)

    def test_apply_filter_history(
        self, chebyshev_filter: ChebyshevFilterDesign, f: np.ndarray
    ):
        k = 1
        component = "gradient"
        p_choice = "L1L"
        chebyshev_filter.apply(
            f=f, component=component, p_choice=p_choice, L=k
        )
        history = chebyshev_filter.history
        assert history is not None

        for _, result in chebyshev_filter.history.items():
            assert result is not None
