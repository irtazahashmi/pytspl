"""Module for edge flow denoising."""

import numpy as np

from pytspl.decomposition.eigendecomposition import get_eigendecomposition
from pytspl.filters.base_filter import BaseFilter
from pytspl.simplicial_complex import SimplicialComplex


class EdgeFlowDenoising(BaseFilter):
    """
    Edge flow denoising with a low-pass filter H_P.

    Solve the regularized optimization probelm to estimate f_tilde from
    f = f0 + ε where, ε is a zero mean Gaussian noise.
    """

    def __init__(self, simplicial_complex: SimplicialComplex):
        """
        Initialize the edge flow denoising filter using the
        simplicial complex.
        """
        super().__init__(simplicial_complex)

    def denoise(
        self,
        f: np.ndarray,
        f_true: np.ndarray,
        p_choice: str,
        mu_vals: np.ndarray = [0.5],
    ):
        """Denoising with low-pass filter H_P.

        Args:
            f (np.ndarray): The noisy signal.
            f_true (np.ndarray): The true signal.
            p_choice (str): The choice of matrix P.
            component (str): The component of the signal.
            mu_vals (np.ndarray, optional): Regularization parameters.
            Defaults to [0.5].
        """
        P = self.get_p_matrix(p_choice).toarray()

        identity = np.eye(P.shape[0])
        U1, _ = get_eigendecomposition(lap_mat=P)

        errors = np.zeros((len(mu_vals)))
        frequency_responses = np.zeros((len(mu_vals), U1.shape[1]))
        f_estimated = None

        # denoising with low pass filter Hp
        for i, mu in enumerate(mu_vals):
            # frequency response of the low-pass filter
            H = np.linalg.inv(identity + mu * P)
            # estimate frequency response
            f_estimated = H @ f
            # calculate error for each mu
            errors[i] = self.calculate_error_NRMSE(f_estimated, f_true)
            # filter frequency response (H_1_tilda)
            frequency_responses[i] = np.diag(U1.T @ H @ U1)

            print(f"mu: {mu}, error: {errors[i]}")

        # update the results
        f_estimated = np.asarray(f_estimated)

        self.set_history(
            filter=H,
            f_estimated=f_estimated,
            frequency_responses=frequency_responses,
            extracted_component_error=errors,
        )

    def plot_desired_frequency_response(self, p_choice: str):
        """Plot the desired frequency response of the filter.

        Args:
            p_choice (str): The choice of matrix P.
        """
        import matplotlib.pyplot as plt

        if self.history["frequency_responses"] is None:
            raise ValueError("Run the denoising method first.")

        P = self.get_p_matrix(p_choice).toarray()
        _, eigenvals = get_eigendecomposition(lap_mat=P)

        frequency_responses = self.history["frequency_responses"]

        plt.figure(figsize=(10, 6))
        plt.plot(eigenvals, frequency_responses[-1])
        plt.xlabel("Frequency", fontsize=12)
        plt.ylabel("Frequency Response", fontsize=12)
        plt.title("Desired Frequency Response", fontsize=14)
