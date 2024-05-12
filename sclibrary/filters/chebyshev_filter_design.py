"""Module for chebyshev filter."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

from chebpy import chebfun
from sclibrary.filters.filter import Filter
from sclibrary.simplicial_complex import SimplicialComplexNetwork
from sclibrary.utils.eigendecomposition import get_eigendecomposition


class ChebyshevFilterDesign(Filter):
    """Chebyshev filter class."""

    def __init__(self, simplicial_complex: SimplicialComplexNetwork):
        """Initialize the Chebyshev filter using the simplicial complex."""
        super().__init__(simplicial_complex=simplicial_complex)

    def _power_iteration(
        self, P: np.ndarray, iterations: int = 50
    ) -> np.ndarray:
        """Power iteration algorithm to approximate the largest eigenvalue.

        Args:
            P (np.ndarray): The input matrix.
            iterations (int): The number of iterations.

        Returns:
            np.ndarray: The approximated largest eigenvalue.
        """
        v = np.ones(P.shape[0])

        for _ in range(iterations):
            v = csr_matrix(P).dot(v)
            v = v / np.linalg.norm(v)

        v = v.astype(float)
        # add small value to avoid division by zero
        v = v + 1e-10
        return v

    def _logistic_function(
        self, cut_off_frequency: float = 0.01, steep: int = 100
    ) -> np.ndarray:
        """
        Compute the logistic function for the given input.

        Args:
            cut_off_frequency (float): The cut-off frequency.
            steep (int): The steepness of the logistic function.

        Returns:
            np.ndarray: The logistic function output.
        """
        return lambda lam: 1 / (1 + np.exp(-steep * (lam - cut_off_frequency)))

    def _get_chebyshev_series(
        self,
        n: int,
        domain_min: float,
        domain_max: float,
        cut_off_frequency: float = 0.01,
        steep: int = 100,
    ) -> np.ndarray:
        """
        Approximate the Chebyshev series for the given points.

        Args:
            n (int): The number of points.
            domain_min (float): The minimum domain value.
            domain_max (float): The maximum domain value.
            cut_off_frequency (float, optional): The cut-off frequency.
            Defaults to 0.01.
            steep (int, optional): The steepness of the logistic function.
            Defaults to 100.

        Returns:
            np.ndarray: The Chebyshev series.
        """

        g_g = self._logistic_function(
            cut_off_frequency=cut_off_frequency, steep=steep
        )
        domain = [domain_min, domain_max]
        g_g_cheb = chebfun(f=g_g, domain=domain, n=n)
        return g_g_cheb

    def _chebyshev_filter_approximate(
        self,
        laplacian_matrix: np.ndarray,
        coefficients: np.ndarray,
        alpha: float,
        k_trnc: int,
    ) -> np.ndarray:
        """
        Approximate the Chebyshev filter.

        Args:
            laplacian_matrix (np.ndarray): The Laplacian matrix.
            coefficients (np.ndarray): The coefficients of the Chebyshev
            filter.
            alpha (float): The alpha value.
            k_trnc (int): The truncation order of the Chebyshev filter.

        Returns:
            np.ndarray: The Chebyshev filter approximation.
        """
        coeffs = np.array(coefficients[:k_trnc])
        K = len(coeffs)

        I = np.eye(laplacian_matrix.shape[0])
        H_cheb_approx = np.zeros(
            (k_trnc, laplacian_matrix.shape[0], laplacian_matrix.shape[1])
        )

        for k in range(K):
            if k == 0:
                H_cheb_approx[k, :, :] = I
            elif k == 1:
                H_cheb_approx[k, :, :] = (
                    1 / alpha * (laplacian_matrix - alpha * I)
                )
            else:
                H_cheb_approx[k, :, :] = (
                    2
                    / alpha
                    * (laplacian_matrix - alpha * I)
                    @ H_cheb_approx[k - 1, :, :]
                    - H_cheb_approx[k - 2, :, :]
                )

        H_cheb_approx_out = np.sum(
            coeffs[:, np.newaxis, np.newaxis] * H_cheb_approx, axis=0
        )
        # multiply coefficients and sum on axis
        return H_cheb_approx_out

    def get_alpha(self, p_choice: str) -> tuple:
        """
        Calculate the alpha value for the given choice of P matrix using
        the power iteration method.

        Args:
            p_choice (str): The choice of P matrix.

        Returns:
            tuple: The alpha and lamda_max value calculated.
        """
        P = self.get_p_matrix(p_choice)
        v = self._power_iteration(P=P)
        # mean of the largest eigenvalue
        lambda_max = np.mean(csr_matrix(P).dot(v) / v)
        # perform a transformation to shit the domain to [0, lambda_g_max]
        alpha = lambda_max / 2
        return alpha, lambda_max

    def get_ideal_frequency(
        self, component_coeffs: np.ndarray, p_choice: str
    ) -> np.ndarray:
        """
        Calculate the ideal frequency of the component.

        Args:
            component coeffs: The masked coefficients of the component.
            p_choice (str): The ideal frequency calculated using the p_choice
            matrix

        Returns:
            np.ndarray: The ideal frequency of the given component and
            p_matrix.
        """

        P = self.get_p_matrix(p_choice)
        U, _ = get_eigendecomposition(lap_mat=P)
        H_ideal = U @ np.diag(component_coeffs) @ U.T
        return H_ideal

    def get_chebyshev_frequency_approx(
        self,
        p_choice: str,
        coeffs: np.ndarray,
        alpha: float,
        k_trunc_order: int,
    ) -> np.ndarray:
        """
        Calculate the Chebyshev frequency approximation.

        Args:
            p_choice (str): The choice of P matrix.
            coeffs (np.ndarray): The coefficients of the Chebyshev filter.
            alpha (float): The alpha value.
            k_trunc_order (int): The truncation order of the Chebyshev
            filter.

        Returns:
            np.ndarray: The Chebyshev frequency approximation.
        """
        P = self.get_p_matrix(p_choice)
        H_cheb_approx = np.zeros((k_trunc_order, P.shape[0], P.shape[1]))
        for k in range(1, k_trunc_order + 1):
            H_cheb_approx[k - 1 :, :, :] = self._chebyshev_filter_approximate(
                laplacian_matrix=P, coefficients=coeffs, alpha=alpha, k_trnc=k
            )
        return H_cheb_approx

    def apply(
        self,
        f: np.ndarray,
        p_choice: str = "L1L",
        component: str = "gradient",
        L: int = 10,
    ) -> np.ndarray:
        """
        Apply the Chebyshev filter to the given flow f.

        Args:
            f (np.ndarray): The input flow.
            p_choice (str, optional): The choice of P matrix.
            Defaults to "L1L".
            component (str, optional): The component of the flow. Defaults
            to "gradient".
            L (int, optional): The filter size. Defaults to 10.

        Returns:
            np.ndarray: The Chebyshev filter output.
        """
        import time

        start = time.time()

        # U, _ = get_eigendecomposition
        # (lap_mat=self.sc.hodge_laplacian_matrix())
        P = self.get_p_matrix(p_choice)
        U_l, _ = get_eigendecomposition(lap_mat=P)
        print(time.time() - start)
        start = time.time()
        print("eigendecomposition done")
        f_true = self.get_true_signal(component=component, f=f)
        h_ideal = self.sc.get_component_coefficients(component=component)
        print(time.time() - start)
        start = time.time()
        print("true signal done")
        # calculate alpha
        alpha, lambda_max = self.get_alpha(p_choice=p_choice)
        print(time.time() - start)
        start = time.time()
        print("alpha done")
        # get the chebyshev coefficients
        g_chebyshev = self._get_chebyshev_series(
            n=len(P), domain_min=0, domain_max=lambda_max
        )
        coeffs = g_chebyshev.funs[0].coeffs
        print(time.time() - start)
        start = time.time()
        print("coeffs done")

        # ideal frequency
        H_ideal = self.get_ideal_frequency(
            p_choice=p_choice, component_coeffs=h_ideal
        )
        print(time.time() - start)
        start = time.time()
        print("ideal frequency done")

        # chebyshev approx frequency
        H_cheb_approx = self.get_chebyshev_frequency_approx(
            p_choice=p_choice,
            coeffs=coeffs,
            alpha=alpha,
            k_trunc_order=L,
        )
        print(time.time() - start)
        start = time.time()
        print("chebyshev approx done")

        errors_response = np.zeros(L)
        errors_filter = np.zeros(L)

        f_cheb = np.zeros((L, P.shape[0]))
        f_cheb_tilde = np.zeros((L, P.shape[0]))

        error_per_filter_size = np.zeros(L)
        # error_tilde = np.zeros(L)

        for k in range(L):
            g_cheb_approx = np.diag(
                U_l.T @ np.squeeze(H_cheb_approx[k, :, :]) @ U_l.T
            )
            # compute the error with respect to the ideal frequency response
            errors_response[k] = self.calculate_error(g_cheb_approx, h_ideal)
            errors_filter[k] = np.linalg.norm(
                np.squeeze(H_cheb_approx[k, :, :]) - H_ideal, 2
            )

            # compute the error with respect to the true signal
            f_cheb[k] = np.squeeze(H_cheb_approx[k, :, :]) @ f
            error_per_filter_size[k] = self.calculate_error(f_cheb[k], f_true)

            # f_tilde - compute the error on component embedding
            # f_cheb_tilde[k] = U.T @ f_cheb[k]
            # error_tilde[k] = self.calculate_error(
            #     f_cheb_tilde[k], U.T @ f_true
            # )

            print(f"Filter size: {k} - Error: {error_per_filter_size[k]}")

        self.history["filter"] = H_cheb_approx.astype(float)
        self.history["f_estimated"] = f_cheb.astype(float)
        self.history["frequency_responses"] = f_cheb_tilde.astype(float)
        self.history["error_per_filter_size"] = error_per_filter_size.astype(
            float
        )

    def plot_chebyshev_series_approx(self, p_choice: str):
        """
        Plot the Chebyshev series approximation.

        Args:
            p_choice (str): The choice of P matrix.
        """
        P = self.get_p_matrix(p_choice)
        _, eigenvals = get_eigendecomposition(lap_mat=P)

        g = self._logistic_function()

        # mean of the largest eigenvalue
        _, lambda_max = self.get_alpha(p_choice=p_choice)
        g_chebysev = self._get_chebyshev_series(
            n=len(P), domain_min=0, domain_max=lambda_max
        )

        plt.figure(figsize=(15, 5))
        # eigenvalues
        plt.scatter(eigenvals, g(eigenvals))
        # chebyshev approx
        plt.scatter(eigenvals, g_chebysev(eigenvals))
        plt.title("Function approximation using Chebyshev polynomials")
        plt.xlabel("Eigenvalues")
        # add legend
        plt.legend(["Function", "Chebyshev approx"])

    def plot_frequency_response_approx(
        self,
        flow: np.ndarray,
        component: str,
    ):
        """
        Plot the frequency response approximation.

        Args:
            flow (np.ndarray): The input flow.
            component (str): The component of the flow.

        Raises:
            ValueError: If the apply method is not run first.
        """

        f_cheb_tilde = self.history["frequency_responses"]
        if f_cheb_tilde is None:
            raise ValueError("Run the apply method first.")

        P = self.get_p_matrix("L1L")
        U, eigenvals = get_eigendecomposition(lap_mat=P)

        f_true = self.get_true_signal(f=flow, component=component)

        plt.figure(figsize=(15, 5))
        plt.scatter(eigenvals, U.T @ f_true)
        plt.scatter(eigenvals, f_cheb_tilde[-1])
        plt.title("Frequency response on the eigenvalues vs chebyshev approx")
        # add legend
        plt.legend(["True flow", "Chebyshev approx"])
