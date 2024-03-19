import numpy as np
from scipy.sparse import csr_matrix

from sclibrary.eigendecomposition import get_eigendecomposition
from sclibrary.simplicial_complex import SimplicialComplexNetwork


class GridBasedFilterDesign:

    def __init__(self, simplicial_complex: SimplicialComplexNetwork):
        self.sc = simplicial_complex
        self.errors = None
        self.frequency_responses = None
        self.f_estimated = None

    def _power_iteration(self, iterations: int = 50) -> np.ndarray:
        """Power iteration algorithm to approximate the largest eigenvalue."""
        L1 = self.sc.hodge_laplacian_matrix()
        v = np.ones(L1.shape[0])

        for i in range(iterations):
            v = csr_matrix(L1).dot(v)
            v = v / np.linalg.norm(v)

        return v

    def _sample_grid_points(self, num_of_samples: int) -> np.ndarray:
        """
        Sample M1 and M2 grid points unoformly in the interval for the smallest
        set value greater than 0 as the lower bound.

        Args:
            num_of_samples (int): Number of samples to take.
        """

        L1 = self.sc.hodge_laplacian_matrix(rank=1)

        # Get the largest eigenvalue
        v = self._power_iteration()
        lambda_min = 0
        lambda_max = np.mean(L1 @ v / v)

        return np.linspace(lambda_min, lambda_max, num_of_samples)

    @staticmethod
    def _frequency_response(eigenvalue: float, mu: float = 0.5) -> float:
        """
        Compute the frequency response for a given eigenvalue.

        Args:
            eigenvalue (float): Eigenvalue of the simplicial complex.
            mu (float): Damping factor.
        """
        return 1 / (1 + mu * eigenvalue)

    def _compute_sampled_continuous_freq_response(
        self, num_of_samples: int, mu: float = 0.5
    ) -> tuple:
        """
        Compute the continuous frequency response for sampled eigenvalues.

        Args:
            num_of_samples (int): Number of samples to take.
            mu (float): Damping factor.
        """

        sampled_eigenvals = self._sample_grid_points(
            num_of_samples=num_of_samples
        )

        # compute the frequency response for each sampled eigenvalue
        g = [
            self._frequency_response(eigenvalue, mu)
            for eigenvalue in sampled_eigenvals
        ]

        return g, sampled_eigenvals

    def _compute_true_continuous_freq_response(self, mu: float = 0.5) -> list:
        """
        Compute the continuous frequency response for the true eigenvalues.

        Args:
            mu (float): Damping factor.
        """
        L1 = self.sc.hodge_laplacian_matrix(rank=1)
        _, eigenvals = get_eigendecomposition(L1)

        # compute the frequency response for each eigenvalue
        g_true = [
            self._frequency_response(eigenvalue, mu)
            for eigenvalue in eigenvals
        ]

        return g_true

    def apply_filter(
        self,
        f0: np.ndarray,
        f: np.ndarray,
        filter_range=range(12),
    ) -> None:

        # eigenvalues
        L1 = self.sc.hodge_laplacian_matrix(rank=1)
        U1, eigenvals = get_eigendecomposition(L1)

        # number of samples
        num_of_samples = len(eigenvals)

        # true eigenvalues & their frequency responses
        g_true = self._compute_true_continuous_freq_response()
        # sample eigenvalues & their frequency responses
        g, eigenvals_sampled = self._compute_sampled_continuous_freq_response(
            num_of_samples=num_of_samples
        )

        # learn the regularization filter with topological filter
        system_mat = np.zeros((len(eigenvals_sampled), len(filter_range)))
        system_mat_true = np.zeros((len(eigenvals), len(filter_range)))

        # errors
        errors_tf = np.zeros((len(filter_range)))
        errors_filter_true = np.zeros((len(filter_range)))

        L1 = np.array(L1, dtype=object)

        for l in filter_range:
            # building the system matrix
            system_mat[:, l] = np.power(eigenvals_sampled, l)
            system_mat_true[:, l] = np.power(eigenvals, l)

            # solve the system using least squares solution to obtain filter coefficients
            h = np.linalg.lstsq(system_mat, g, rcond=None)[0]
            h_true = np.linalg.lstsq(system_mat_true, g_true, rcond=None)[0]

            # build the topology filter
            H = np.zeros_like(L1, dtype=object)
            H_true = np.zeros_like(L1, dtype=object)

            for i in range(len(h)):
                H += h[i] * np.linalg.matrix_power(L1, i)
                H_true += h_true[i] * np.linalg.matrix_power(L1, i)

            # estimate the signal
            f_est = H @ f

            # frequency response of the filter
            H_freq = np.diag(U1.T @ H @ U1)

            # compute errors
            errors_tf[l] = np.linalg.norm(f_est - f0) / np.linalg.norm(f0)
            # H - g_G(lambda)
            errors_filter_true[l] = np.linalg.norm(H - H_true)

        self.errors = errors_filter_true.astype(float)
        self.frequency_responses = H_freq.astype(float)
        self.f_estimated = f_est.astype(float)
