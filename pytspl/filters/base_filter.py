"""Filter module for filter design and denoising."""

import numpy as np
from scipy.sparse import csr_matrix

from pytspl.decomposition.frequency_component import FrequencyComponent
from pytspl.simplicial_complex import SimplicialComplex


class BaseFilter:
    """Base filter class for desining linear filters."""

    def __init__(self, simplicial_complex: SimplicialComplex):
        """Initialize the filter design using a simplicial complex.

        The history of the filter design is stored in the history attribute.
        """
        self.sc = simplicial_complex

        self.history = {
            "filter": None,
            "f_estimated": None,
            "frequency_responses": None,
            "extracted_component_error": None,
            "filter_error": None,
        }

    def _reset_history(self):
        """Reset the history of the filter design."""
        self.history = {
            "filter": None,
            "f_estimated": None,
            "extracted_component_error": None,
            "filter_error": None,
        }

    def set_history(
        self,
        filter: np.ndarray,
        f_estimated: np.ndarray,
        frequency_responses: np.ndarray,
        extracted_component_error: np.ndarray,
        filter_error: np.ndarray = np.array([]),
    ) -> None:
        """Set the history of the filter design."""
        self.history["filter"] = filter.astype(float)
        self.history["f_estimated"] = f_estimated.astype(float)
        self.history["frequency_responses"] = frequency_responses.astype(float)
        self.history["extracted_component_error"] = (
            extracted_component_error.astype(float)
        )
        self.history["filter_error"] = filter_error.astype(float)

    @staticmethod
    def calculate_error_NRMSE(f_estimated: np.ndarray, f_true) -> float:
        """Calculate the error of the estimated signal using NRMSE."""
        return np.linalg.norm(f_estimated - f_true) / np.linalg.norm(f_true)

    @staticmethod
    def power_iteration(P: np.ndarray, iterations: int = 50) -> np.ndarray:
        """Power iteration algorithm to approximate the largest eigenvalue.

        Args:
            P (np.ndarray): The input matrix.
            iterations (int): The number of iterations.

        Returns:
            np.ndarray: The approximated largest eigenvalue.
        """
        v = np.ones(P.shape[0])

        for _ in range(iterations):
            v = P @ v
            v = v / np.linalg.norm(v)

        v = v.astype(float)
        return v

    def get_true_signal(self, f: np.ndarray, component: str) -> np.ndarray:
        """
        Get the true signal for the component.

        Args:
            f (np.ndarray): The signal to be filtered.
            component (str): The component to be extracted.

        Returns:
            np.ndarray: The true signal.
        """
        f_true = self.sc.get_component_flow(
            flow=f, component=component, round_fig=False
        )
        return f_true

    def get_p_matrix(self, p_choice: str = "L1") -> csr_matrix:
        """
        Get the matrix P for the filter design. The matrix P can be
        the Laplacian matrix, the lower Laplacian matrix, or the upper
        Laplacian matrix.

        Args:
            p_choice (str, optional): The choice of matrix P. Defaults
            to "L1". Choose from ['L1', 'L1L', 'L1U'].

        Raises:
            ValueError: Invalid P_choice.

        Returns:
            csr_matrix: The matrix P.
        """
        P_choices = {
            "L1": self.sc.hodge_laplacian_matrix(rank=1),
            "L1L": self.sc.lower_laplacian_matrix(rank=1),
            "L1U": self.sc.upper_laplacian_matrix(rank=1),
        }

        try:
            P = P_choices[p_choice]
        except KeyError:
            raise ValueError(
                "Invalid P_choice. Choose from ['L1', 'L1L', 'L1U']"
            )

        return P

    def get_component_coefficients(
        self,
        component: str,
    ) -> np.ndarray:
        """
        Calculate the component coefficients of the given component using the
        order of the eigenvectors.

        Args:
            component (str): Component of the eigendecomposition to return.

        Raises:
            ValueError: If the component is not one of 'harmonic', 'curl',
            or 'gradient'.

        Returns:
            np.ndarray: The component coefficients of the simplicial complex
            for the given component.
        """
        L1 = self.sc.hodge_laplacian_matrix(rank=1).toarray()

        U_H, e_h = self.sc.get_component_eigenpair(
            FrequencyComponent.HARMONIC.value
        )
        U_C, e_c = self.sc.get_component_eigenpair(
            component=FrequencyComponent.CURL.value
        )
        _, e_g = self.sc.get_component_eigenpair(
            component=FrequencyComponent.GRADIENT.value
        )

        # concatenate the eigenvalues
        eigenvals = np.concatenate((e_h, e_c, e_g))

        # mask the eigenvectors
        mask = np.zeros(L1.shape[0])

        if component == FrequencyComponent.HARMONIC.value:
            mask[: U_H.shape[1]] = 1
        elif component == FrequencyComponent.CURL.value:
            mask[U_H.shape[1] : U_H.shape[1] + U_C.shape[1]] = 1
        elif component == FrequencyComponent.GRADIENT.value:
            mask[U_H.shape[1] + U_C.shape[1] :] = 1
        else:
            raise ValueError(
                "Invalid component. Choose from 'harmonic', 'curl', "
                + "or 'gradient'."
            )

        # sort mask according to eigenvalues
        mask = mask[np.argsort(eigenvals)]

        return mask
