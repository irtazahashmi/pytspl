import numpy as np
import pytest

from sclibrary import SimplicialComplexNetwork
from sclibrary.filters import LSFilterDesign


@pytest.fixture(autouse=True)
def ls_filter(sc: SimplicialComplexNetwork):
    return LSFilterDesign(sc)


class TestLSFilterDesign:

    def test_subcomponent_extraction_type_one_gradient(
        self, ls_filter: LSFilterDesign, f: np.ndarray
    ):
        filter_size = 4
        component = "gradient"

        ls_filter.subcomponent_extraction_type_one(
            L=filter_size, component=component, f=f
        )

        # none of the attributes should be None for the history
        for _, result in ls_filter.history.items():
            assert result is not None

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

    def test_subcomponent_extraction_type_one_gradient_large_filter_order(
        self, ls_filter: LSFilterDesign, f: np.ndarray
    ):
        filter_size = 16
        component = "gradient"

        ls_filter.subcomponent_extraction_type_one(
            L=filter_size, component=component, f=f
        )

        # none of the attributes should be None for the history
        for _, result in ls_filter.history.items():
            assert result is not None

        # test error is decreasing
        assert np.all(
            np.diff(ls_filter.history["error_per_filter_size"]) < 0.1
        )

        # test the estimated signal
        f_expected = np.array(
            [2.48, 0.56, 1.89, -1.92, 1.33, 1.84, 1.01, -0.51, 0.95, 1.45]
        )
        assert np.allclose(
            np.round(ls_filter.history["f_estimated"], 2),
            f_expected,
        )

        expected_error = 0.0009
        actual_error = ls_filter.history["error_per_filter_size"]
        assert np.isclose(actual_error[-1], expected_error, atol=1e-4)

    def test_subcomponent_extraction_type_one_gradient_error(
        self, ls_filter: LSFilterDesign, f: np.ndarray
    ):
        filter_size = 4
        component = "gradient"

        ls_filter.subcomponent_extraction_type_one(
            L=filter_size, component=component, f=f
        )

        expected_error = 0.39
        actual_error = ls_filter.calculate_error(
            ls_filter.history["f_estimated"], f
        )
        assert np.isclose(actual_error, expected_error, atol=1e-2)

    def test_subcomponent_extraction_type_two_gradient(
        self, ls_filter: LSFilterDesign, f: np.ndarray
    ):
        filter_size = 4
        component = "gradient"

        ls_filter.subcomponent_extraction_type_two(
            L=filter_size, component=component, f=f
        )

        # none of the attributes should be None for the history
        for _, result in ls_filter.history.items():
            assert result is not None

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

    def test_subcomponent_extraction_type_two_gradient_large_filter_order(
        self, ls_filter: LSFilterDesign, f: np.ndarray
    ):
        filter_size = 16
        component = "gradient"

        ls_filter.subcomponent_extraction_type_two(
            L=filter_size, component=component, f=f
        )

        # none of the attributes should be None for the history
        for _, result in ls_filter.history.items():
            assert result is not None

        # test error is decreasing
        assert np.all(
            np.diff(ls_filter.history["error_per_filter_size"]) < 0.1
        )

        expected_error = 0.0058
        actual_error = ls_filter.history["error_per_filter_size"]
        assert np.isclose(actual_error[-1], expected_error, atol=1e-4)

    def test_subcomponent_extraction_type_two_component_error(
        self, ls_filter: LSFilterDesign, f: np.ndarray
    ):
        filter_size = 4
        component = "unknown"

        with pytest.raises(ValueError):
            ls_filter.subcomponent_extraction_type_two(
                L=filter_size, component=component, f=f
            )

    def test_history_subcomponent_extraction_type_one(
        self, ls_filter: LSFilterDesign, f: np.ndarray
    ):
        filter_size = 4
        component = "gradient"

        ls_filter.subcomponent_extraction_type_one(
            L=filter_size, component=component, f=f
        )

        assert ls_filter.history["filter"] is not None
        assert isinstance(ls_filter.history["filter"], np.ndarray)

        assert ls_filter.history["f_estimated"] is not None
        assert isinstance(ls_filter.history["f_estimated"], np.ndarray)

        assert ls_filter.history["frequency_responses"] is not None
        assert isinstance(ls_filter.history["frequency_responses"], np.ndarray)

        assert ls_filter.history["error_per_filter_size"] is not None
        assert isinstance(
            ls_filter.history["error_per_filter_size"], np.ndarray
        )

    def test_general_filter(self, ls_filter: LSFilterDesign, f: np.ndarray):

        L1, L2 = 1, 1
        f_est_h, f_est_c, f_est_g = ls_filter.general_filter(L1=L1, L2=L2, f=f)
        f_est = f_est_h + f_est_c + f_est_g

        assert ls_filter.history["L1"] is not None
        assert ls_filter.history["L2"] is not None

        assert np.allclose(
            np.round(f_est, 2),
            f,
        )

    def test_history_general_filter(
        self, ls_filter: LSFilterDesign, f: np.ndarray
    ):
        L1, L2 = 1, 1
        ls_filter.general_filter(L1=L1, L2=L2, f=f)

        assert ls_filter.history["L1"] is not None
        assert ls_filter.history["L2"] is not None
