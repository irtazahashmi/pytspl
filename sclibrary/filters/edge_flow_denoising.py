import numpy as np
from scipy.sparse import csr_matrix

from sclibrary.filters.filter import Filter
from sclibrary.simplicial_complex import SimplicialComplexNetwork
from sclibrary.utils.eigendecomposition import get_eigendecomposition

"""Module for edge flow denoising."""


class EdgeFlowDenoising(Filter):
    """
    Solve the regularized optimization probelm to estimate f_tilde from
    f = f0 + ε where, ε is a zero mean Gaussian noise.
    """

    def __init__(self, simplicial_complex: SimplicialComplexNetwork):
        super().__init__(simplicial_complex)

    def denoise(
        self,
        p_choice: str,
        component: str,
        f: np.ndarray,
        mu_vals: np.ndarray = [0.5],
    ):
        """
        Denoising with low-pass filter Hp.

        Args:
            p_choice (str): The choice of matrix P.
            component (str): The component of the signal.
            f (np.ndarray): The noisy signal.
            mu_vals (np.ndarray, optional): Regularization parameters.
            Defaults to [0.5].

        """

        P = self.get_p_matrix(p_choice)

        identity = np.eye(P.shape[0])
        U1, _ = get_eigendecomposition(P)
        f_true = self.get_true_signal(component=component, f=f)

        errors = np.zeros((len(mu_vals)))
        frequency_responses = np.zeros((len(mu_vals), U1.shape[1]))

        # denoising with low pass filter Hp
        for i, mu in enumerate(mu_vals):
            # frequency response of the low-pass filter
            H = np.linalg.inv(identity + mu * P)

            # estimate frequency response
            f_estimated = csr_matrix(H, dtype=float).dot(f)

            # calculate error for each mu
            errors[i] = self.calculate_error(f_estimated, f_true)

            # filter frequency response (H_1_tilda)
            frequency_responses[i] = np.diag(U1.T @ H @ U1)

        # update the results
        self.history["filter"] = H
        self.history["f_estimated"] = np.array(f_estimated).astype(float)
        self.history["frequency_responses"] = np.array(
            frequency_responses
        ).astype(float)
        self.history["error_per_filter_size"] = np.array(errors).astype(float)
