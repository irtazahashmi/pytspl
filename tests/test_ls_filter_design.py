import numpy as np
import pytest

from sclibrary.ls_filter_design import LSFilterDesign
from sclibrary.simplicial_complex import SimplicialComplexNetwork


@pytest.fixture
def f():
    return np.array(
        [2.90, 0.25, 1.78, -1.50, 1.76, 1.53, 1.32, 0.08, 0.67, 1.73]
    )


@pytest.fixture(autouse=True)
def ls_filter(sc: SimplicialComplexNetwork):
    return LSFilterDesign(sc)


class TestLSFilterDesign:

    def test_get_true_signal_gradient(
        self, ls_filter: LSFilterDesign, f: np.ndarray
    ):
        component = "gradient"
        f_true = ls_filter._get_true_signal(component, f)

        f_expected = np.array(
            [2.48, 0.56, 1.89, -1.92, 1.34, 1.84, 1.01, -0.51, 0.95, 1.45]
        )
        assert np.allclose(
            np.round(f_true, 2),
            f_expected,
        )

    def test_get_true_signal_error(
        self, ls_filter: LSFilterDesign, f: np.ndarray
    ):
        component = "unknown"

        with pytest.raises(ValueError):
            ls_filter._get_true_signal(component, f)

    def test_simplicial_filter_gradient(
        self, ls_filter: LSFilterDesign, f: np.ndarray
    ):

        filter_size = 4
        component = "gradient"

        ls_filter.simplicial_filter(L=filter_size, component=component, f=f)

        assert len(ls_filter.history["error_per_filter_size"]) != 0
        assert len(ls_filter.history["frequency_responses"]) != 0
        assert ls_filter.history["f_estimated"] is not None

        # test error is decreasing

        assert np.all(
            np.diff(ls_filter.history["error_per_filter_size"]) < 0.1
        )

        f_expected = np.array(
            [1.86, -0.09, 1.11, -1.25, 1.36, 1.37, 0.25, 0.32, 0.51, 0.85]
        )

        assert np.allclose(
            np.round(ls_filter.history["f_estimated"], 2),
            f_expected,
        )

    def test_simplicial_filter_gradient_error(
        self, ls_filter: LSFilterDesign, f: np.ndarray
    ):

        filter_size = 4
        component = "gradient"

        ls_filter.simplicial_filter(L=filter_size, component=component, f=f)

        expected_error = 0.39
        assert np.isclose(
            ls_filter.history["error"], expected_error, atol=1e-2
        )

    def test_subcomponent_extraction_gradient(
        self, ls_filter: LSFilterDesign, f: np.ndarray
    ):

        filter_size = 4
        component = "gradient"

        ls_filter.subcomponent_extraction(
            L=filter_size, component=component, f=f
        )

        assert len(ls_filter.history["error_per_filter_size"]) != 0
        assert len(ls_filter.history["frequency_responses"]) != 0
        assert ls_filter.history["f_estimated"] is not None

        # test error is decreasing
        assert np.all(
            np.diff(ls_filter.history["error_per_filter_size"]) < 0.1
        )

        f_expected = np.array(
            [2.62, 0.34, 1.56, -2.19, 1.24, 1.69, 0.94, -0.36, 0.94, 1.4]
        )

        assert np.allclose(
            np.round(ls_filter.history["f_estimated"], 2),
            f_expected,
        )

    def test_subcomponent_extraction_error(
        self, ls_filter: LSFilterDesign, f: np.ndarray
    ):

        filter_size = 4
        component = "unknown"

        with pytest.raises(ValueError):
            ls_filter.subcomponent_extraction(
                L=filter_size, component=component, f=f
            )

    def test_general_filter(self, ls_filter: LSFilterDesign, f: np.ndarray):

        L1, L2 = 1, 1
        ls_filter.general_filter(L1=L1, L2=L2, f=f)

        assert np.allclose(
            np.round(ls_filter.history["f_estimated"], 2),
            f,
        )
