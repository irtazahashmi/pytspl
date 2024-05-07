import numpy as np
from scipy.sparse import csr_matrix

from sclibrary.filters.filter import Filter
from sclibrary.simplicial_complex import SimplicialComplexNetwork
from sclibrary.utils.eigendecomposition import get_eigendecomposition
from sclibrary.utils.frequency_component import FrequencyComponent


class LSFilterDesign(Filter):
    """Module for the LS filter design."""

    def __init__(self, simplicial_complex: SimplicialComplexNetwork):
        """Initialize the LS filter design using a simplicial complex."""
        super().__init__(simplicial_complex)

    def _apply_filter(
        self,
        L: int,
        lap_matrix: np.ndarray,
        f: np.ndarray,
        f_true: np.ndarray,
        U: np.ndarray,
        eigenvals: np.ndarray,
        alpha: np.ndarray,
    ) -> tuple:
        """
        Apply the filter to the signal using a type of the laplacian matrix.

        Args:
            L (int): The size of the filter.
            lap_matrix (np.ndarray): The laplacian matrix. It can be the Hodge
            Laplacian matrix, the upper or lower laplacian matrix.
            f (np.ndarray): The noisy signal to be filtered.
            f_true (np.ndarray): The true signal.
            U (np.ndarray): The eigenvectors of the laplacian matrix.
            eigenvals (np.ndarray): The eigenvalues of the laplacian matrix.
            alpha (np.ndarray): The coefficients of the filter.

        Returns:
            tuple: The estimated filter, signal, frequency responses and
            error per filter size.
        """
        # convert L1 to a sparse matrix
        L1 = csr_matrix(lap_matrix, dtype=float)

        # create a matrix to store the system
        system_mat = np.zeros((len(eigenvals), L))

        # store the results
        errors = np.zeros((L))
        frequency_responses = np.zeros((L, len(U)))
        f_estimated = None

        for L in range(L):
            # create the system matrix
            system_mat[:, L] = np.power(eigenvals, L)

            # Least square solution to obtain the filter coefficients
            h = np.linalg.lstsq(system_mat, alpha, rcond=None)[0]

            # building the topological filter
            H = np.zeros_like(L1, dtype=float)

            for l in range(len(h)):
                H += h[l] * csr_matrix(L1**l, dtype=float)

            # filter the signal
            f_estimated = csr_matrix(H, dtype=float).dot(f)

            # compute the error for each filter size
            errors[L] = self.calculate_error(f_estimated, f_true)

            # filter frequency response (H_1_tilda)
            frequency_responses[L] = np.diag(U.T @ H @ U)

            print(f"Filter size: {L} - Error: {errors[L]}")

        f_estimated = np.array(f_estimated).astype(float)
        frequency_responses = np.array(frequency_responses).astype(float)
        errors = np.array(errors).astype(float)

        return H, f_estimated, frequency_responses, errors

    def subcomponent_extraction_type_one(
        self, L: int, component: str, f: np.ndarray
    ) -> None:
        """
        LS based filter design for subcomponent extraction using the Hodge
        Laplacian matrix (L1) - type one.

        In this case, we will use the Hodge Laplacian matrix L1 = L2 = L
        and α = β.

        Hk = sum(l=0, L) h_l * L^l

        Args:
            L (int): The size of the filter.
            component (str): The component to be extracted.
            f (np.ndarray): The signal to be filtered.
        """
        self._reset_history()

        # eigendecomposition of the Hodge Laplacian matrix
        L1 = self.sc.hodge_laplacian_matrix(rank=1)
        U, eigenvals = get_eigendecomposition(lap_mat=L1)

        # get the true signal
        f_true = self.get_true_signal(component=component, f=f)

        # get the component coefficients
        alpha = self.sc.get_component_coefficients(component=component)

        H, f_estimated, frequency_responses, errors = self._apply_filter(
            L=L,
            lap_matrix=L1,
            f=f,
            f_true=f_true,
            U=U,
            eigenvals=eigenvals,
            alpha=alpha,
        )

        # update the results
        self.history["filter"] = H
        self.history["f_estimated"] = f_estimated
        self.history["frequency_responses"] = frequency_responses
        self.history["error_per_filter_size"] = errors

    def subcomponent_extraction_type_two(
        self,
        L: int,
        component: str,
        f: np.ndarray,
        tolerance: float = 1e-6,
    ) -> None:
        """
        LS based filter design for subcomponent extraction using the upper or
        lower Laplacian matrix (L1 or L2) - type two.

        In this case:
        - The solution will have zero coefficients on the α for curl
        extraction (L1 = 0)
        - The solution will have zero coefficients on the β for gradient
        extraction (L2 = 0)

        Therefore, we will only consider the upper or lower part of the filter
        to do so.

        Args:
            L (int): The size of the filter.
            component (str): The component to be extracted.
            f (np.ndarray): The signal to be filtered.
            tolerance (float, optional): The tolerance to consider the
            eigenvalues as unique. Defaults to 1e-6.
        """
        self._reset_history()

        # get the Laplacian matrix according to the component
        componenet_mapping = {
            FrequencyComponent.GRADIENT.value: self.sc.lower_laplacian_matrix(
                rank=1
            ),
            FrequencyComponent.CURL.value: self.sc.upper_laplacian_matrix(
                rank=1
            ),
        }
        try:
            lap_matrix = componenet_mapping[component]
        except KeyError:
            raise ValueError(
                f"Invalid component {component}. Use 'gradient' or 'curl'."
            )

        # get the eigenvalues
        U, eigenvals = get_eigendecomposition(
            lap_mat=lap_matrix, tolerance=tolerance
        )
        # unique eigenvalues
        eigenvals = np.unique(eigenvals)

        # get the true signal
        f_true = self.get_true_signal(component=component, f=f)

        # filter coefficients
        alpha = [0] + [1] * (len(eigenvals) - 1)

        # apply the filter
        H, f_estimated, frequency_responses, errors = self._apply_filter(
            L=L,
            lap_matrix=lap_matrix,
            f=f,
            f_true=f_true,
            U=U,
            eigenvals=eigenvals,
            alpha=alpha,
        )

        # update the results
        self.history["filter"] = H
        self.history["f_estimated"] = f_estimated
        self.history["frequency_responses"] = frequency_responses
        self.history["error_per_filter_size"] = errors

    def general_filter(
        self,
        L1: int,
        L2: int,
        f: np.ndarray,
        tolerance: float = 1e-6,
    ) -> np.ndarray:
        """
        Denoising by a general filter H1 with L1 != L2 = L and α != β.

        Args:
            L1 (int): The size of the filter for the gradient extraction.
            L2 (int): The size of the filter for the curl extraction.
            f (np.ndarray): The signal to be filtered.
            tolerance (float, optional): The tolerance to consider the
            eigenvalues as unique. Defaults to 1e-6.

        Returns:
            np.ndarray: The estimated harmonic, curl and gradient components.
        """
        self._reset_history()

        f_est_g, f_est_c, f_est_h = 0, 0, 0

        history = {
            "L1": None,
            "L2": None,
        }

        # gradient extraction
        if L1 > 0:
            self.subcomponent_extraction_type_two(
                L=L1,
                component=FrequencyComponent.GRADIENT.value,
                f=f,
                tolerance=tolerance,
            )
            f_est_g = self.history["f_estimated"]
            history["L1"] = self.history

        # curl extraction
        if L2 > 0:
            self.subcomponent_extraction_type_two(
                L=L2,
                component=FrequencyComponent.CURL.value,
                f=f,
                tolerance=tolerance,
            )
            f_est_c = self.history["f_estimated"]
            history["L2"] = self.history

        # harmonic extraction
        f_est_h = f - f_est_g - f_est_c

        # update history
        self.history = history

        return f_est_h, f_est_c, f_est_g
