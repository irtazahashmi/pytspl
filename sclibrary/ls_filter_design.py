import matplotlib.pyplot as plt
import numpy as np

from sclibrary.eigendecomposition import get_eigendecomposition
from sclibrary.freq_component import FrequencyComponent
from sclibrary.simplicial_complex import SimplicialComplexNetwork


class LSFilterDesign:

    def __init__(self, sc: SimplicialComplexNetwork):
        self.sc = sc
        self.errors = None
        self.frequency_responses = None
        self.f_estimated = None

    def _get_true_signal(self, component: str, f: np.ndarray) -> np.ndarray:
        """
        Get the true signal for the component.

        Args:
            component (str): The component to be extracted.
            f (np.ndarray): The signal to be filtered.

        Returns:
            np.ndarray: The true signal.
        """
        f_h, f_c, f_g = self.sc.get_hodgedecomposition(flow=f, round_fig=False)
        component_mapping = {
            FrequencyComponent.HARMONIC.value: f_h,
            FrequencyComponent.CURL.value: f_c,
            FrequencyComponent.GRADIENT.value: f_g,
        }

        f_true = component_mapping[component]
        if f_true is None:
            raise ValueError(
                f"Invalid component {component}. Use 'harmonic', 'curl' or 'gradient'."
            )

        return f_true

    def _apply_filter(
        self,
        lap_matrix: np.ndarray,
        f: np.ndarray,
        f_true: np.ndarray,
        u_1: np.ndarray,
        eigenvals: np.ndarray,
        alpha: np.ndarray,
        filter_range: np.ndarray,
    ) -> np.ndarray:

        # convert L1 to a numpy array with dtype object
        L1 = np.array(lap_matrix, dtype=object)

        # create a matrix to store the system
        system_mat = np.zeros((len(eigenvals), len(filter_range)))
        f_estimated = []
        errors = []
        frequency_responses = np.zeros((len(u_1), len(filter_range)))

        for L in filter_range:

            system_mat[:, L] = np.power(eigenvals, L)

            # Least square solution to obtain the filter coefficients
            h = np.linalg.lstsq(system_mat, alpha, rcond=None)[0]

            # building the topological filter
            H_1 = np.zeros_like(L1, dtype=object)

            for l in range(len(h)):
                H_1 += h[l] * np.linalg.matrix_power(L1, l)

            # filter the signal
            f_estimated = H_1 @ f

            # compute the error
            error = np.linalg.norm(f_estimated - f_true) / np.linalg.norm(
                f_true
            )
            errors.append(error)

            # filter frequency response
            H_1_tilda = np.diag(u_1.T @ H_1 @ u_1)
            frequency_responses[:, L] = H_1_tilda

        self.f_estimated = f_estimated
        self.errors = errors
        self.frequency_responses = frequency_responses

    def general_filter(
        self,
        component: str,
        f: np.ndarray,
        filter_range: np.ndarray = range(12),
    ) -> tuple:
        """
        Direct LS based filter design for subcomponent extraction by filter H1.
        In this case, we will use the Hodge Laplacian matrix L1 = L2 = L and α = β.

        Args:
            component (str): The component to be extracted.
            f (np.ndarray): The signal to be filtered.
            filter_range (np.ndarray, optional): The range of the size of the filters to be tested.
            Defaults to range(12).

        Returns:
            tuple: The estimated signal, errors and frequency responses.
        """

        # eigendecomposition of the Hodge Laplacian matrix
        L1 = self.sc.hodge_laplacian_matrix(rank=1)
        u_1, eigenvals = get_eigendecomposition(L1)

        # get the component coefficients
        alpha = self.sc.get_component_coefficients(component=component)
        # get the true signal
        f_true = self._get_true_signal(component, f)

        self._apply_filter(
            lap_matrix=L1,
            f=f,
            f_true=f_true,
            u_1=u_1,
            eigenvals=eigenvals,
            alpha=alpha,
            filter_range=filter_range,
        )

    def subcomponent_extraction(
        self,
        component: str,
        f: np.ndarray,
        filter_range: np.ndarray = range(12),
        tolerance: float = 1e-6,
    ) -> tuple:
        """
        Direct LS based filter design for subcomponent extraction by filter H1.


        In this case:
        - The solution will have zero coefficients on the α for curl extraction (L1 = 0)
        - The solution will have zero coefficients on the β for gradient extraction (L2 = 0)

        Therefore, we will only consider the upper or lower part of the filter
        to do so.

        Args:
            component (str): The component to be extracted.
            f (np.ndarray): The signal to be filtered.
            filter_range (np.ndarray, optional): The range of the size of the filters to
            be tested. Defaults to range(12).
            tolerance (float, optional): The tolerance to consider the eigenvalues as unique.
            Defaults to 1e-6.

        Returns:
            tuple: The estimated signal, errors and frequency responses.
        """

        # get the Laplacian matrix according to the component
        componenet_mapping = {
            FrequencyComponent.CURL.value: self.sc.upper_laplacian_matrix(
                rank=1
            ),
            FrequencyComponent.GRADIENT.value: self.sc.lower_laplacian_matrix(
                rank=1
            ),
        }
        lap_matrix = componenet_mapping[component]
        if lap_matrix is None:
            raise ValueError(
                f"Invalid component {component}. Use 'harmonic' or 'curl'."
            )

        # get unique eigenvalues
        u_1, eigenvals = get_eigendecomposition(
            lap_matrix, tolerance=tolerance
        )
        eigenvals = np.unique(eigenvals)
        # get the true signal
        f_true = self._get_true_signal(component, f)
        # filter coefficients
        alpha = [0] + [1] * (len(eigenvals) - 1)

        self._apply_filter(
            lap_matrix=lap_matrix,
            f=f,
            f_true=f_true,
            u_1=u_1,
            eigenvals=eigenvals,
            alpha=alpha,
            filter_range=filter_range,
        )
