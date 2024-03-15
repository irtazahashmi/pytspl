import numpy as np

from sclibrary.eigendecomposition import get_eigendecomposition
from sclibrary.simplicial_complex import SimplicialComplexNetwork


class EdgeFlowDenoising:
    """
    Solve the regularized optimization probelm to estimate f_tilde from f = f0 + ε
    where, ε is a zero mean Gaussian noise.
    """

    def __init__(self, simplicial_complex: SimplicialComplexNetwork):
        self.sc = simplicial_complex
        self.errors = None
        self.f_estimated = None

    def denoise(
        self,
        f0: np.ndarray,
        f: np.ndarray,
        mu_vals: np.ndarray = [0.5],
        P_choice="L1",
    ):
        """
        Denoising with low-pass filter Hp.

        Args:
            f0 (np.ndarray): The noise-free flow.
            f (np.ndarray): The noisy flow.
            mu_vals (np.ndarray, optional): Regularization parameters.
            Defaults to [0.5].
            P (str, optional): The choice of matrix P. Defaults to "L1".
        """

        if P_choice == "L1":
            P = self.sc.hodge_laplacian_matrix(rank=1)
        elif P_choice == "L1L":
            P = self.sc.lower_laplacian_matrix(rank=1)
        else:
            raise ValueError("Invalid P_choice. Choose from ['L1', 'L1L']")

        I = np.eye(P.shape[0])
        U1, _ = get_eigendecomposition(P)

        # Use P as the edge laplacian
        A_lg = np.abs(P - 2 * I)
        L_lg = np.diag(A_lg) - A_lg

        # regularization parameters
        errors = np.zeros((len(mu_vals)))

        for i, mu in enumerate(mu_vals):
            # denoising with low pass filter Hp

            # frequency response of the low-pass filter
            H_regularized = np.linalg.inv(P * mu + I)
            H_regularized_freq = np.diag(U1.T @ H_regularized @ U1)

            f_est_r = H_regularized @ f

            # calculate error
            errors[i] = np.linalg.norm(f_est_r - f0) / np.linalg.norm(f0)

        self.errors = errors
        self.f_estimated = f_est_r
