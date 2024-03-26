import numpy as np

from sclibrary.eigendecomposition import get_eigendecomposition
from sclibrary.simplicial_complex import SimplicialComplexNetwork


class EdgeFlowDenoising:
    """
    Solve the regularized optimization probelm to estimate f_tilde from
    f = f0 + ε where, ε is a zero mean Gaussian noise.
    """

    def __init__(self, simplicial_complex: SimplicialComplexNetwork):
        self.sc = simplicial_complex

        self.f_estimated = None
        self.error = None

        self.filter_range = None
        self.errors = None

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

        errors = np.zeros((len(mu_vals)))

        # denoising with low pass filter Hp
        for i, mu in enumerate(mu_vals):
            # frequency response of the low-pass filter
            H = np.linalg.inv(I + mu * P)
            H_freq = np.diag(U1.T @ H @ U1)

            # estimate frequency response
            f_estimated = H @ f

            # calculate error for each mu
            errors[i] = np.linalg.norm(f_estimated - f0) / np.linalg.norm(f0)

        self.f_estimated = f_estimated
        self.error = np.linalg.norm(f_estimated - f0) / np.linalg.norm(f0)

        self.filter_range = mu_vals
        self.errors = errors
