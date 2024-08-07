"""Grid-based filter design."""

import numpy as np
from scipy.sparse import csr_matrix

from pytspl.decomposition.eigendecomposition import get_eigendecomposition
from pytspl.filters.base_filter import BaseFilter
from pytspl.simplicial_complex import SimplicialComplex


class GridBasedFilterDesign(BaseFilter):
    """Module for grid-based filter design inheriting from the
    BaseFilter class.
    """

    def __init__(self, simplicial_complex: SimplicialComplex):
        """
        Initialize the grid-based filter design using the simplicial
        complex.
        """
        super().__init__(simplicial_complex)

    def _sample_grid_points(
        self, P: np.ndarray, num_of_samples: int
    ) -> np.ndarray:
        """
        Sample M1 and M2 grid points uniformly in the interval for the smallest
        set value greater than 0 as the lower bound.

        Args:
            P (np.ndarray): The matrix P.
            num_of_samples (int): Number of samples to take.

        Returns:
            np.ndarray: Sampled grid points.
        """
        # Get the largest eigenvalue
        v = self.power_iteration(P=P)
        lambda_min = 0
        lambda_max = np.mean(P @ v / v)
        return np.linspace(lambda_min, lambda_max, num_of_samples)

    @staticmethod
    def _compute_frequency_response_hp(
        eigenvalue: float, mu: float = 0.5
    ) -> float:
        """
        Compute the frequency response of the low-pass filter (Hp)
        for the given eigenvalue.

        Args:
            eigenvalue (float): The eigenvalue.
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
            np.ndarray: Sampled frequency responses.
            np.ndarray: Sampled eigenvalues.
        """
        sampled_eigenvals = self._sample_grid_points(
            P=P, num_of_samples=num_of_samples
        )

        # compute the frequency response for each sampled eigenvalue
        sampled_freq_response = [
            self._compute_frequency_response_hp(eigenvalue=eigenvalue, mu=mu)
            for eigenvalue in sampled_eigenvals
        ]

        sampled_freq_response = np.asarray(sampled_freq_response, dtype=float)
        return sampled_freq_response, sampled_eigenvals

    def _apply_filter(
        self,
        f: np.ndarray,
        f_true: np.ndarray,
        P: csr_matrix,
        alpha: np.ndarray,
        U: np.ndarray,
        eigenvals: np.ndarray,
        L: int,
    ) -> tuple:
        """
        Apply the filter to the input signal.

        Args:
            f (np.ndarray): The input signal.
            f_true (np.ndarray): The true signal.
            P (csr_matrix): The matrix P.
            alpha (np.ndarray): The filter coefficients.
            U (np.ndarray): The eigenvectors.
            eigenvals (np.ndarray): The eigenvalues.
            L (int): The filter size.

        Returns:
            np.ndarray: The filter.
            np.ndarray: The estimated signal.
            np.ndarray: The frequency responses.
            np.ndarray: The errors in the extracted component.
        """
        # learn the regularization filter with topological filter
        system_mat = np.zeros((len(eigenvals), L))

        f_estimated = None
        # errors
        errors = np.zeros((L))
        # frequency responses
        frequency_responses = np.zeros((L, len(U)))

        for l in range(L):
            # building the system matrix
            system_mat[:, l] = eigenvals**l

            # solve the system using least squares solution to obtain
            # filter coefficients
            h = np.linalg.lstsq(system_mat, alpha, rcond=None)[0]

            # build the topology filter
            H = np.zeros(P.shape, dtype=float)

            for i in range(len(h)):
                H += h[i] * (P**i).toarray()

            # estimate the signal
            f_estimated = H @ f
            # compute error compared to the true component signal
            errors[l] = self.calculate_error_NRMSE(f_estimated, f_true)
            # frequency response of the filter
            frequency_responses[l] = np.diag(U.T @ H @ U)

            print(f"Filter size: {l} - Error: {np.round(errors[l], 4)}")

        return H, f_estimated, frequency_responses, errors

    def denoise(
        self,
        f: np.ndarray,
        f_true: np.ndarray,
        p_choice: str,
        L: int,
        mu: float = 0.5,
    ) -> None:
        """
        Build a low-pass filter H_P to denoise the input signal.

        Args:
            f (np.ndarray): The noisy signal.
            f_true (np.ndarray): The true signal.
            p_choice (str): The choice of matrix P.
            L (int): The filter size.
            mu (float): The damping factor.
        """
        P = self.get_p_matrix(p_choice)
        U, _ = get_eigendecomposition(lap_mat=P.toarray())

        # number of samples
        num_of_samples = len(f_true)
        # sample eigenvalues & their frequency responses
        g, eigenvals_sampled = self._compute_sampled_continuous_freq_response(
            P=P.toarray(), num_of_samples=num_of_samples, mu=mu
        )

        H, f_estimated, frequency_responses, errors = self._apply_filter(
            f=f,
            f_true=f_true,
            P=P,
            alpha=g,
            U=U,
            eigenvals=eigenvals_sampled,
            L=L,
        )

        # update the results
        self.set_history(
            filter=H,
            f_estimated=f_estimated,
            frequency_responses=frequency_responses,
            extracted_component_error=errors,
        )

    def subcomponent_extraction(
        self,
        f: np.ndarray,
        component: str,
        p_choice: str,
        L: int,
        num_of_samples: int = 10,
        mu: float = 0.5,
        cut_off_frequency: float = 0.01,
        steep: int = 100,
    ) -> None:
        """
        Subcomponent extraction using the grid-based simplicial filter H_1.

        Args:
            f (np.ndarray): The noisy signal.
            component (str): The component to extract.
            p_choice (str): The choice of matrix P.
            L (int): The filter size.
            num_of_samples (int, Optional): Number of eigenvalues to
            sample. Default is 10.
            mu (float, Optional): The damping factor. Default is 0.5.
            cut_off_frequency (float, Optional): The cut-off frequency.
            Default is 0.01.
            steep (int, Optional): The steepness of the logistic function.
            Default is 100.
        """
        P = self.get_p_matrix(p_choice)
        U, _ = get_eigendecomposition(lap_mat=P.toarray())

        # sample eigenvalues & their frequency responses
        _, eigenvals_sampled = self._compute_sampled_continuous_freq_response(
            P=P.toarray(), num_of_samples=num_of_samples, mu=mu
        )

        g_g = self.logistic_function(
            cut_off_frequency=cut_off_frequency, steep=steep
        )
        alpha = g_g(eigenvals_sampled)

        f_true = self.get_true_signal(f=f, component=component)

        H, f_estimated, frequency_responses, errors = self._apply_filter(
            P=P,
            alpha=alpha,
            f=f,
            f_true=f_true,
            U=U,
            eigenvals=eigenvals_sampled,
            L=L,
        )

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
    ) -> tuple:
        """
        Apply a general filter H1 with L1 != L2 = L and α != β.

        Args:
            f (np.ndarray): The signal to be filtered.
            L1 (int): The size of the filter for the gradient extraction.
            L2 (int): The size of the filter for the curl extraction.

        Returns:
            np.ndarray: The estimated harmonic component.
            np.ndarray: The estimated curl component.
            np.ndarray: The estimated gradient component.
        """
        f_est_g, f_est_c, f_est_h = 0, 0, 0

        history = {
            "L1": None,
            "L2": None,
        }

        # gradient extraction
        if L1 > 0:
            self.subcomponent_extraction(
                f=f, component="gradient", p_choice="L1L", L=L1
            )
            f_est_g = self.history["f_estimated"]
            history["L1"] = self.history

        # curl extraction
        if L2 > 0:
            self.subcomponent_extraction(
                f=f, component="curl", p_choice="L1U", L=L2
            )
            f_est_c = self.history["f_estimated"]
            history["L2"] = self.history

        # harmonic extraction
        f_est_h = f - f_est_g - f_est_c

        # update history
        self.history = history

        return f_est_h, f_est_c, f_est_g
