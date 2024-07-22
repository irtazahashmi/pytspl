import numpy as np
import pytest

from pytspl import SimplicialComplex
from pytspl.filters import ChebyshevFilterDesign


@pytest.fixture(autouse=True)
def chebyshev_filter(sc_mock: SimplicialComplex):
    return ChebyshevFilterDesign(sc_mock)


@pytest.fixture
def cheb_filter_chicago(sc_chicago_mock: SimplicialComplex):
    return ChebyshevFilterDesign(sc_chicago_mock)


@pytest.fixture
def f():
    return np.array(
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


@pytest.fixture
def coeffs(chebyshev_filter: ChebyshevFilterDesign):
    n = 10
    domain_min = 0
    _, domain_max = chebyshev_filter.get_alpha(p_choice="L1L")
    g_cheb = chebyshev_filter._get_chebyshev_series(
        n=n, domain_min=domain_min, domain_max=domain_max
    )
    coeffs = g_cheb.funs[0].coeffs
    yield coeffs


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
        coeffs: np.ndarray,
    ):
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
        assert np.allclose(coeffs, expected_output)

    def test_chebyshev_filter_approximate(
        self, chebyshev_filter: ChebyshevFilterDesign, coeffs: np.ndarray
    ):
        L1L = chebyshev_filter.sc.lower_laplacian_matrix(rank=1).toarray()

        alpha_g = 2.74

        result = chebyshev_filter._chebyshev_filter_approximate(
            P=L1L, coefficients=coeffs, alpha=alpha_g, k_trnc=1
        )
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, np.eye(len(L1L)) * 0.9593, atol=1e-4)

    def test_get_chebyshev_frequency_approx(
        self, chebyshev_filter: ChebyshevFilterDesign, coeffs: np.ndarray
    ):

        k, alpha_g = 2, 2.74
        g_cheb = chebyshev_filter.get_chebyshev_frequency_approx(
            p_choice="L1L", coeffs=coeffs, alpha=alpha_g, k_trunc_order=k
        )
        assert isinstance(g_cheb, np.ndarray)

        expected_first = np.array(
            [
                0.93744792,
                0.02964554,
                0.02964554,
                -0.02964554,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        )
        excepted_second = np.array(
            [
                0.02964554,
                0.93744792,
                0.02964554,
                0.02964554,
                -0.02964554,
                -0.02964554,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        )

        assert np.allclose(g_cheb[:, :, 0][0], expected_first, atol=1e-8)
        assert np.allclose(g_cheb[:, :, 1][0], excepted_second, atol=1e-8)

    def test_get_alpha(self, chebyshev_filter: ChebyshevFilterDesign):
        alpha, lambda_max = chebyshev_filter.get_alpha(p_choice="L1L")
        expected_alpha, expected_lambda_max = 2.74, 5.49
        assert np.round(alpha, 2) == expected_alpha
        assert np.round(lambda_max, 2) == expected_lambda_max

    def test_apply_filter_order_10(
        self, chebyshev_filter: ChebyshevFilterDesign, f: np.ndarray
    ):

        k = 10
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

    def test_apply_filter_order_200(
        self, chebyshev_filter: ChebyshevFilterDesign, f: np.ndarray
    ):
        k, n = 200, 200
        cut_off_frequency = 0.1

        component = "gradient"
        p_choice = "L1L"
        chebyshev_filter.apply(
            f=f,
            component=component,
            p_choice=p_choice,
            L=k,
            n=n,
            cut_off_frequency=cut_off_frequency,
        )

        actual_error = chebyshev_filter.history["extracted_component_error"][
            -1
        ]
        expected_error = 4e-05

        assert actual_error < expected_error

    def test_apply_filter_order_30_chicago(
        self,
        cheb_filter_chicago: ChebyshevFilterDesign,
        f0_chicago_mock: np.ndarray,
    ):
        k, n = 30, 100
        cut_off_frequency = 0.01

        component = "gradient"
        p_choice = "L1L"
        cheb_filter_chicago.apply(
            f=f0_chicago_mock,
            component=component,
            p_choice=p_choice,
            L=k,
            n=n,
            cut_off_frequency=cut_off_frequency,
        )

        actual_comp_error = cheb_filter_chicago.history[
            "extracted_component_error"
        ][-1]
        expected_comp_error = 0.2
        assert actual_comp_error < expected_comp_error

        actual_filter_error = cheb_filter_chicago.history["filter_error"][-1]
        expected_filter_error = 0.4
        assert actual_filter_error < expected_filter_error

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

    def test_plot_chebyshev_series_approx(
        self, chebyshev_filter: ChebyshevFilterDesign
    ):
        import matplotlib.pyplot as plt

        # Call the plot_chebyshev_series_approx method
        chebyshev_filter.plot_chebyshev_series_approx(p_choice="L1L", n=10)
        # Check if the plot is displayed
        assert plt.gcf().number == 1

    def test_plot_frequency_response_approx(
        self, chebyshev_filter: ChebyshevFilterDesign, f: np.ndarray
    ):
        import matplotlib.pyplot as plt

        k, n = 10, 10
        component = "gradient"
        p_choice = "L1L"
        chebyshev_filter.apply(
            f=f,
            component=component,
            p_choice=p_choice,
            L=k,
            n=n,
        )

        # Call the plot_chebyshev_series_approx method
        chebyshev_filter.plot_frequency_response_approx(
            flow=f, component=component
        )
        # Check if the plot is displayed
        assert plt.gcf().number == 2

    def test_plot_plot_frequency_response_approx_without_running_filter(
        self, chebyshev_filter: ChebyshevFilterDesign, f: np.ndarray
    ):
        # catch ValueError when the filter is not applied
        with pytest.raises(ValueError):
            chebyshev_filter.plot_frequency_response_approx(
                flow=f, component="gradient"
            )
