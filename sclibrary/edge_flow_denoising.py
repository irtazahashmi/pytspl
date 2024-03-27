import numpy as np
from scipy.sparse import csr_matrix

from sclibrary.eigendecomposition import get_eigendecomposition
from sclibrary.simplicial_complex import SimplicialComplexNetwork


class EdgeFlowDenoising:
    """
    Solve the regularized optimization probelm to estimate f_tilde from
    f = f0 + ε where, ε is a zero mean Gaussian noise.
    """

    def __init__(self, simplicial_complex: SimplicialComplexNetwork):
        self.sc = simplicial_complex

        self.history = {
            "f_estimated": None,
            "error": None,
            "frequency_responses": None,
            "error_per_filter_size": None,
        }

    def calculate_error(self, f_estimated: np.ndarray, f_true) -> float:
        """
        Calculate the error of the estimated signal.
        """
        return np.linalg.norm(f_estimated - f_true) / np.linalg.norm(f_true)

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

        P_choices = {
            "L1": self.sc.hodge_laplacian_matrix(rank=1),
            "L1L": self.sc.lower_laplacian_matrix(rank=1),
            "L1U": self.sc.upper_laplacian_matrix(rank=1),
        }

        try:
            P = P_choices[P_choice]
        except KeyError:
            raise ValueError(
                "Invalid P_choice. Choose from ['L1', 'L1L', 'L1U']"
            )

        I = np.eye(P.shape[0])
        U1, _ = get_eigendecomposition(P)

        errors = np.zeros((len(mu_vals)))
        frequency_responses = np.zeros((len(mu_vals), U1.shape[1]))

        # denoising with low pass filter Hp
        for i, mu in enumerate(mu_vals):
            # frequency response of the low-pass filter
            H = np.linalg.inv(I + mu * P)

            # estimate frequency response
            f_estimated = csr_matrix(H, dtype=float).dot(f)

            # calculate error for each mu
            errors[i] = self.calculate_error(f_estimated, f0)

            # filter frequency response (H_1_tilda)
            frequency_responses[i] = np.diag(U1.T @ H @ U1)

        f_estimated = np.array(f_estimated).astype(float)
        frequency_responses = np.array(frequency_responses).astype(float)
        errors = np.array(errors).astype(float)

        # update the results
        self.history["f_estimated"] = f_estimated
        self.history["error"] = self.calculate_error(f_estimated, f0)
        self.history["frequency_responses"] = frequency_responses
        self.history["error_per_filter_size"] = errors
