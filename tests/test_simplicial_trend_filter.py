import numpy as np
import pytest

from pytspl import load_dataset
from pytspl.filters import SimplicialTrendFilter


@pytest.fixture(autouse=True)
def trend_filter():
    sc, _, _ = load_dataset("lastfm-1k-artist")
    trend_filter = SimplicialTrendFilter(simplicial_complex=sc)
    yield trend_filter


@pytest.fixture(autouse=True)
def f():
    _, _, flow = load_dataset("lastfm-1k-artist")
    flow = np.asarray(list(flow.values()))
    yield flow


class TestSimplicialTrendFilter:

    def test_set_history(self, trend_filter: SimplicialTrendFilter):
        # Set the history
        filter = np.array([1, 2, 3])
        frequency_responses = np.array([[1, 2, 3], [4, 5, 6]])
        component_flow = np.array([1, 2])
        errors = np.array([0.1, 0.2])
        correlations = np.array([0.1, 0.2])

        trend_filter.set_history(
            filter, frequency_responses, component_flow, errors, correlations
        )

        # Check if the history is set correctly
        assert np.array_equal(trend_filter.history["filter"], filter)
        assert np.array_equal(
            trend_filter.history["frequency_responses"],
            frequency_responses,
        )
        assert np.array_equal(
            trend_filter.history["component_flow"], component_flow
        )
        assert np.array_equal(trend_filter.history["errors"], errors)
        assert np.array_equal(
            trend_filter.history["correlations"], correlations
        )

    def test_denoising_l2_regularizer(
        self, trend_filter: SimplicialTrendFilter, f: np.ndarray
    ):
        component = "divergence"
        num_realizations = 50
        snr_db = np.arange(-12, 12.5, 12)

        trend_filter.denoising_l2_regularizer(
            flow=f,
            component=component,
            num_realizations=num_realizations,
            snr_db=snr_db,
        )
        expected_error = 0.003
        assert trend_filter.history["errors"][-1] < expected_error

        expected_corr = 0.998
        assert trend_filter.history["correlations"][-1] > expected_corr

    def test_denoising_l1_regularizer(
        self, trend_filter: SimplicialTrendFilter, f: np.ndarray
    ):
        component = "divergence"
        order = 0
        num_realizations = 50
        snr_db = np.arange(-12, 12.5, 12)

        trend_filter.denoising_l1_regularizer(
            flow=f,
            order=order,
            component=component,
            num_realizations=num_realizations,
            snr_db=snr_db,
        )
        expected_error = 0.004
        assert trend_filter.history["errors"][-1] < expected_error

        expected_corr = 0.998
        assert trend_filter.history["correlations"][-1] > expected_corr

    def test_interpolation_l1_regularizer(
        self, trend_filter: SimplicialTrendFilter, f: np.ndarray
    ):
        order = 0
        component = "divergence"
        ratio = np.arange(0.05, 1.05, 0.3)
        num_realizations = 50

        trend_filter.interpolation_l1_regularizer(
            flow=f,
            order=order,
            component=component,
            ratio=ratio,
            num_realizations=num_realizations,
        )
        expected_error = 0.020
        assert trend_filter.history["errors"][-1] < expected_error

        expected_corr = 0.999
        assert trend_filter.history["correlations"][-1] > expected_corr
