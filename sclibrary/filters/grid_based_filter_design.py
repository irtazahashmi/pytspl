import numpy as np
from scipy.sparse import csr_matrix

from sclibrary.filters.filter import Filter
from sclibrary.simplicial_complex import SimplicialComplexNetwork
from sclibrary.utils.eigendecomposition import get_eigendecomposition
from sclibrary.utils.frequency_component import FrequencyComponent

"""Module for grid-based filter design."""


class GridBasedFilterDesign(Filter):

    def __init__(self, simplicial_complex: SimplicialComplexNetwork):
        super().__init__(simplicial_complex)

    def _power_iteration(
        self, P: np.ndarray, iterations: int = 50
    ) -> np.ndarray:
        """Power iteration algorithm to approximate the largest eigenvalue."""
        v = np.ones(P.shape[0])

        for _ in range(iterations):
            v = csr_matrix(P).dot(v)
            v = v / np.linalg.norm(v)

        v = v.astype(float)
        # add small value to avoid division by zero
        v = v + 1e-10
        return v

    def _sample_grid_points(
        self, P: np.ndarray, num_of_samples: int
    ) -> np.ndarray:
        """
        Sample M1 and M2 grid points unoformly in the interval for the smallest
        set value greater than 0 as the lower bound.

        Args:
            P (np.ndarray): The matrix P.
            num_of_samples (int): Number of samples to take.

        Returns:
            np.ndarray: Sampled grid points.
        """

        # Get the largest eigenvalue
        v = self._power_iteration(P=P)
        lambda_min = 0
        lambda_max = np.mean(csr_matrix(P).dot(v) / v)
        return np.linspace(lambda_min, lambda_max, num_of_samples)

    @staticmethod
    def _compute_frequency_response(
        eigenvalue: float, mu: float = 0.5
    ) -> float:
        """
        Compute the frequency response for a given eigenvalue.

        Args:
            eigenvalue (float): Eigenvalue of the simplicial complex.
            mu (float): Damping factor.

        Returns:
            float: Frequency response.
        """
        return 1 / (1 + mu * eigenvalue)

    def _compute_sampled_continuous_freq_response(
        self, P: np.ndarray, num_of_samples: int, mu: float = 0.5
    ) -> tuple:
        """
        Compute the continuous frequency response for sampled eigenvalues.

        Args:
            P (np.ndarray): The matrix P.
            num_of_samples (int): Number of samples to take.
            mu (float): Damping factor.

        Returns:
            tuple: Sampled frequency responses and sampled eigenvalues.
        """

        sampled_eigenvals = self._sample_grid_points(
            P=P, num_of_samples=num_of_samples
        )

        # compute the frequency response for each sampled eigenvalue
        sampled_freq_response = [
            self._compute_frequency_response(eigenvalue, mu)
            for eigenvalue in sampled_eigenvals
        ]

        return sampled_freq_response, sampled_eigenvals

    def _compute_true_continuous_freq_response(
        self, P: np.ndarray, mu: float = 0.5
    ) -> list:
        """
        Compute the continuous frequency response for the true eigenvalues.

        Args:
            P (np.ndarray): The matrix P.
            mu (float): Damping factor.

        Returns:
            list: True frequency responses.
        """
        _, eigenvals = get_eigendecomposition(P)

        # compute the frequency response for each eigenvalue
        g_true = [
            self._compute_frequency_response(eigenvalue, mu)
            for eigenvalue in eigenvals
        ]

        return g_true

    def subcomponent_extraction(
        self,
        p_choice: str,
        L: int,
        component: str,
        f: np.ndarray,
    ) -> None:
        """
        Apply the grid-based filter to the input signal.

        Args:
            p_choice (str): The choice of matrix P.
            L (int): The filter size.
            component (str): The component of the signal.
            f (np.ndarray): The noisy signal.
        """

        P = self.get_p_matrix(p_choice)

        U1, eigenvals = get_eigendecomposition(lap_mat=P)
        f_true = self.get_true_signal(component=component, f=f)

        # number of samples
        num_of_samples = len(eigenvals)

        # true eigenvalues & their frequency responses
        g_true = self._compute_true_continuous_freq_response(P=P)
        # sample eigenvalues & their frequency responses
        g, eigenvals_sampled = self._compute_sampled_continuous_freq_response(
            P=P, num_of_samples=num_of_samples
        )

        # learn the regularization filter with topological filter
        system_mat = np.zeros((len(eigenvals_sampled), L))
        system_mat_true = np.zeros((len(eigenvals), L))

        # errors
        errors = np.zeros((L))
        errors_per_filter_size = np.zeros((L))

        for l in range(L):

            # building the system matrix
            system_mat[:, l] = np.power(eigenvals_sampled, l)
            system_mat_true[:, l] = np.power(eigenvals, l)

            # solve the system using least squares solution to obtain
            # filter coefficients
            h = np.linalg.lstsq(system_mat, g, rcond=None)[0]
            h_true = np.linalg.lstsq(system_mat_true, g_true, rcond=None)[0]

            # build the topology filter
            H = np.zeros_like(P, dtype=object)
            H_true = np.zeros_like(P, dtype=object)

            for i in range(len(h)):
                H += h[i] * np.linalg.matrix_power(P, i)
                H_true += h_true[i] * np.linalg.matrix_power(P, i)

            # estimate the signal
            f_est = csr_matrix(H, dtype=float).dot(f)

            # frequency response of the filter
            frequency_responses = np.diag(U1.T @ H @ U1)

            # compute error compared to the true component signal
            errors[l] = self.calculate_error(f_est, f_true)
            # computer error compared to the true filter using the
            # true eigenvalues
            errors_per_filter_size[l] = np.linalg.norm(H - H_true)

        # update the results
        self.history["filter"] = H
        self.history["f_estimated"] = f_est.astype(float)
        self.history["frequency_responses"] = frequency_responses.astype(float)
        self.history["error_per_filter_size"] = errors_per_filter_size.astype(
            float
        )
        self.history["errors"] = errors.astype(float)

    def general_filter(
        self,
        L1: int,
        L2: int,
        f: np.ndarray,
    ) -> np.ndarray:
        """
        Denoising by a general filter H1 with L1 != L2 = L and α != β.

        Args:
            L1 (int): The size of the filter for the gradient extraction.
            L2 (int): The size of the filter for the curl extraction.
            f (np.ndarray): The signal to be filtered.

        Returns:
            np.ndarray: The estimated harmonic, curl and gradient components.
        """

        f_est_g, f_est_c, f_est_h = 0, 0, 0

        history = {
            "L1": None,
            "L2": None,
        }

        # gradient extraction
        if L1 > 0:
            self.subcomponent_extraction(
                p_choice="L1L",
                L=L1,
                component=FrequencyComponent.GRADIENT.value,
                f=f,
            )
            f_est_g = self.history["f_estimated"]
            history["L1"] = self.history

        # curl extraction
        if L2 > 0:
            self.subcomponent_extraction(
                p_choice="L1U",
                L=L2,
                component=FrequencyComponent.CURL.value,
                f=f,
            )
            f_est_c = self.history["f_estimated"]
            history["L2"] = self.history

        # harmonic extraction
        f_est_h = f - f_est_g - f_est_c

        # update history
        self.history = history

        return f_est_h, f_est_c, f_est_g