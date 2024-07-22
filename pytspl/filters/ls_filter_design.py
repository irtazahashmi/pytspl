"""LS-based filter design module for subcomponent extraction of type I
and type II filters.
"""

import numpy as np
from scipy.sparse import csr_matrix

from pytspl.decomposition.eigendecomposition import get_eigendecomposition
from pytspl.decomposition.frequency_component import FrequencyComponent
from pytspl.filters.base_filter import BaseFilter
from pytspl.simplicial_complex import SimplicialComplex


class LSFilterDesign(BaseFilter):
    """Module for LS filter design inheriting from the BaseFilter class."""

    def __init__(self, simplicial_complex: SimplicialComplex):
        """Initialize the LS filter design using a simplicial complex."""
        super().__init__(simplicial_complex)

    def _apply_filter(
        self,
        f: np.ndarray,
        f_true: np.ndarray,
        lap_matrix: csr_matrix,
        U: np.ndarray,
        eigenvals: np.ndarray,
        alpha: np.ndarray,
        L: int,
    ) -> tuple:
        """
        Apply the filter to the signal using a type of the laplacian matrix.

        Args:
            f (np.ndarray): The noisy signal to be filtered.
            f_true (np.ndarray): The true signal.
            lap_matrix (csr_matrix): The laplacian matrix. It can be the Hodge
            Laplacian matrix, the upper or lower laplacian matrix.
            U (np.ndarray): The eigenvectors of the laplacian matrix.
            eigenvals (np.ndarray): The eigenvalues of the laplacian matrix.
            alpha (np.ndarray): The coefficients of the filter.
            L (int): The size of the filter.

        Returns:
            tuple: The estimated filter, signal, frequency responses and
            error per filter size.
        """
        # create a matrix to store the system
        system_mat = np.zeros((len(eigenvals), L))

        # store the results
        errors = np.zeros((L))
        frequency_responses = np.zeros((L, len(U)))
        f_estimated = None

        for L in range(L):
            # create the system matrix
            if L == 0:
                system_mat[:, L] = np.ones(len(eigenvals))
            else:
                system_mat[:, L] = system_mat[:, L - 1] * eigenvals

            # Least square solution to obtain the filter coefficients
            h = np.linalg.lstsq(system_mat, alpha, rcond=None)[0]

            # building the topological filter
            H = np.zeros(lap_matrix.shape, dtype=float)

            for l in range(len(h)):
                H += h[l] * (lap_matrix**l).toarray()

            # filter the signal
            f_estimated = H @ f

            # compute the error for each filter size
            errors[L] = self.calculate_error_NRMSE(f_estimated, f_true)

            # filter frequency response (H_1_tilda)
            frequency_responses[L] = np.diag(U.T @ H @ U)

            print(f"Filter size: {L} - Error: {errors[L]}")

        f_estimated = np.asarray(f_estimated)

        return H, f_estimated, frequency_responses, errors

    def subcomponent_extraction_type_one(
        self,
        f: np.ndarray,
        component: str,
        L: int,
    ) -> None:
        """
        LS based filter design for subcomponent extraction using the Hodge
        Laplacian matrix (L1) - type one.

        In this case, we will use the Hodge Laplacian matrix L1 = L2 = L
        and α = β.

        Hk = sum(l=0, L) h_l * L^l

        Args:
            f (np.ndarray): The signal to be filtered.
            component (str): The component to be extracted.
            L (int): The size of the filter.
        """
        self._reset_history()

        # eigendecomposition of the Hodge Laplacian matrix
        L1 = self.sc.hodge_laplacian_matrix(rank=1)
        U, eigenvals = get_eigendecomposition(lap_mat=L1.toarray())

        # get the true signal
        f_true = self.get_true_signal(component=component, f=f)

        # get the component coefficients
        alpha = self.get_component_coefficients(component=component)

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
        self.set_history(
            filter=H,
            f_estimated=f_estimated,
            frequency_responses=frequency_responses,
            extracted_component_error=errors,
        )

    def subcomponent_extraction_type_two(
        self,
        f: np.ndarray,
        component: str,
        L: int,
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
            f (np.ndarray): The signal to be filtered.
            L (int): The size of the filter.
            component (str): The component to be extracted.
            tolerance (float, optional): The tolerance to consider the
            eigenvalues as unique. Defaults to 1e-6.
        """
        self._reset_history()

        # get the Laplacian matrix
        if component == FrequencyComponent.GRADIENT.value:
            lap_matrix = self.sc.lower_laplacian_matrix(rank=1)
        elif component == FrequencyComponent.CURL.value:
            lap_matrix = self.sc.upper_laplacian_matrix(rank=1)
        else:
            raise ValueError(
                f"Invalid component {component}. Use 'gradient' or 'curl'."
            )

        # get the unique eigenvalues
        U, eigenvals = get_eigendecomposition(lap_mat=lap_matrix.toarray())
        eigenvals = np.where(np.abs(eigenvals) < tolerance, 0, eigenvals)
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
        # update the results
        self.set_history(
            filter=H,
            f_estimated=f_estimated,
            frequency_responses=frequency_responses,
            extracted_component_error=errors,
        )

    def general_filter(
        self,
        f: np.ndarray,
        L1: int,
        L2: int,
        tolerance: float = 1e-3,
    ) -> np.ndarray:
        """
        Denoising by a general filter H1 with L1 != L2 = L and α != β.

        Args:
            f (np.ndarray): The signal to be filtered.
            L1 (int): The size of the filter for the gradient extraction.
            L2 (int): The size of the filter for the curl extraction.
            tolerance (float, optional): The tolerance to consider the
            eigenvalues as unique. Defaults to 1e-3.

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
                f=f,
                component=FrequencyComponent.GRADIENT.value,
                L=L1,
                tolerance=tolerance,
            )
            f_est_g = self.history["f_estimated"]
            history["L1"] = self.history

        # curl extraction
        if L2 > 0:
            self.subcomponent_extraction_type_two(
                f=f,
                component=FrequencyComponent.CURL.value,
                L=L2,
                tolerance=tolerance,
            )
            f_est_c = self.history["f_estimated"]
            history["L2"] = self.history

        # harmonic extraction
        f_est_h = f - f_est_g - f_est_c

        # update history
        self.history = history

        return f_est_h, f_est_c, f_est_g
