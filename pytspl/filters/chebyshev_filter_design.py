"""Module for Chebyshev filter design."""

import matplotlib.pyplot as plt
import numpy as np

from chebpy import chebfun
from pytspl.decomposition.eigendecomposition import get_eigendecomposition
from pytspl.filters.base_filter import BaseFilter
from pytspl.simplicial_complex import SimplicialComplex


class ChebyshevFilterDesign(BaseFilter):
    """Chebyshev filter design inheriting from the BaseFilter class."""

    def __init__(self, simplicial_complex: SimplicialComplex):
        """Initialize the Chebyshev filter using the simplicial complex."""
        super().__init__(simplicial_complex=simplicial_complex)

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
        P: np.ndarray,
        coefficients: np.ndarray,
        alpha: float,
        k_trnc: int,
    ) -> np.ndarray:
        """
        Approximate the Chebyshev filter.

        Args:
            P (np.ndarray): The Laplacian matrix.
            coefficients (np.ndarray): The coefficients of the Chebyshev
            filter.
            alpha (float): The alpha value.
            k_trnc (int): The truncation order of the Chebyshev filter.

        Returns:
            np.ndarray: The Chebyshev filter approximation.
        """
        coeffs = np.asarray(coefficients[:k_trnc])
        K = len(coeffs)

        I = np.eye(P.shape[0], P.shape[1])
        H_cheb_approx = np.zeros((k_trnc, P.shape[0], P.shape[1]), dtype=float)

        for k in range(K):
            if k == 0:
                H_cheb_approx[k, :, :] = I
            elif k == 1:
                H_cheb_approx[k, :, :] = 1 / alpha * (P - alpha * I)
            else:
                H_cheb_approx[k, :, :] = (
                    2 / alpha * (P - alpha * I) @ H_cheb_approx[k - 1, :, :]
                    - H_cheb_approx[k - 2, :, :]
                )

        # combine all approximations using the coefficients
        H_cheb_approx_out = np.sum(
            coeffs[:, np.newaxis, np.newaxis] * H_cheb_approx, axis=0
        )
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
        P = self.get_p_matrix(p_choice).toarray()
        v = self.power_iteration(P=P)

        # mean of the largest eigenvalue
        lambda_max = np.mean((P @ v) / v)
        # perform a transformation to shit the domain to [0, lambda_g_max]
        alpha = lambda_max / 2
        return alpha, lambda_max

    def get_ideal_frequency(self, component_coeffs: np.ndarray) -> np.ndarray:
        """
        Calculate the ideal frequency of the component.

        Args:
            component coeffs: The masked coefficients of the component.

        Returns:
            np.ndarray: The ideal frequency of the given component and
            p_matrix.
        """
        P = self.get_p_matrix("L1").toarray()
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
        P = self.get_p_matrix(p_choice).toarray()

        H_cheb_approx = np.zeros(
            (k_trunc_order, P.shape[0], P.shape[1]), dtype=float
        )

        for k in range(k_trunc_order):
            print(f"Calculating Chebyshev filter approximation for k = {k}...")
            H_cheb_approx[k - 1 :, :, :] = self._chebyshev_filter_approximate(
                P=P, coefficients=coeffs, alpha=alpha, k_trnc=k + 1
            )

        return H_cheb_approx

    def apply(
        self,
        f: np.ndarray,
        p_choice: str = "L1L",
        component: str = "gradient",
        L: int = 10,
        n: int = None,
        cut_off_frequency: float = 0.01,
        steep: int = 100,
    ) -> None:
        """
        Apply the Chebyshev filter to the given flow f.

        Args:
            f (np.ndarray): The input flow.
            p_choice (str, optional): The choice of P matrix.
            Defaults to "L1L".
            component (str, optional): The component of the flow. Defaults
            to "gradient".
            L (int, optional): The filter size. Defaults to 10.
            n (int, optional): The number of points. Defaults to None.
            cut_off_frequency (float, optional): The cut-off frequency.
            Defaults to 0.01.
            steep (int, optional): The steepness of the logistic function.
            Defaults to 100.
        """
        # if n is not provided, the filter size is the same as the
        # number of points
        if not n:
            n = L

        L1 = self.sc.hodge_laplacian_matrix().toarray()
        U, _ = get_eigendecomposition(lap_mat=L1)

        P = self.get_p_matrix(p_choice).toarray()
        U_l, _ = get_eigendecomposition(lap_mat=P)

        f_true = self.get_true_signal(f=f, component=component)
        h_ideal = self.get_component_coefficients(component=component)

        # calculate alpha
        alpha, lambda_max = self.get_alpha(p_choice=p_choice)

        # get the chebyshev coefficients
        g_chebyshev = self._get_chebyshev_series(
            n=n,
            domain_min=0,
            domain_max=lambda_max,
            cut_off_frequency=cut_off_frequency,
            steep=steep,
        )
        coeffs = g_chebyshev.funs[0].coeffs

        # ideal frequency
        H_ideal = self.get_ideal_frequency(component_coeffs=h_ideal)

        # chebyshev approx frequency
        H_cheb_approx = self.get_chebyshev_frequency_approx(
            p_choice=p_choice,
            coeffs=coeffs,
            alpha=alpha,
            k_trunc_order=L,
        )

        errors_response = np.zeros(L)
        errors_filter = np.zeros(L)

        f_cheb = np.zeros((L, P.shape[0]))
        f_cheb_tilde = np.zeros((L, P.shape[0]))

        extracted_comp_error = np.zeros(L)
        error_tilde = np.zeros(L)

        for k in range(L):
            g_cheb_approx = np.diag(
                U_l.T @ np.squeeze(H_cheb_approx[k, :, :]) @ U_l
            )
            # compute the error with respect to the true component extraction
            errors_response[k] = self.calculate_error_NRMSE(
                g_cheb_approx, h_ideal
            )
            errors_filter[k] = np.linalg.norm(
                np.squeeze(H_cheb_approx[k, :, :]) - H_ideal, ord=2
            )

            # compute the error with respect to the true signal
            f_cheb[k] = np.squeeze(H_cheb_approx[k, :, :]) @ f
            extracted_comp_error[k] = self.calculate_error_NRMSE(
                f_cheb[k], f_true
            )

            # f_tilde - compute the error on component embedding
            f_cheb_tilde[k] = U.T @ f_cheb[k]
            error_tilde[k] = self.calculate_error_NRMSE(
                f_cheb_tilde[k], U.T @ f_true
            )

            print(
                f"Filter size: {k} - Error: {extracted_comp_error[k]} - "
                + f"Filter error: {errors_filter[k]} - "
                + f"Error response: {errors_response[k]}"
            )

        self.set_history(
            filter=H_cheb_approx,
            f_estimated=f_cheb,
            frequency_responses=f_cheb_tilde,
            extracted_component_error=extracted_comp_error,
            filter_error=errors_filter,
        )

    def plot_chebyshev_series_approx(
        self,
        p_choice: str,
        n: int = None,
        cut_off_frequency: float = 0.01,
        steep: int = 100,
    ) -> None:
        """
        Plot the Chebyshev series approximation.

        Args:
            p_choice (str): The choice of P matrix.
            n (int, optional): The number of points. Defaults to None.
            cut_off_frequency (float, optional): The cut-off frequency.
            Defaults to 0.01.
            steep (int, optional): The steepness of the logistic function.
            Defaults to 100.
        """
        P = self.get_p_matrix(p_choice).toarray()
        _, eigenvals = get_eigendecomposition(lap_mat=P)

        if not n:
            n = len(P)

        g = self._logistic_function()

        # mean of the largest eigenvalue
        _, lambda_max = self.get_alpha(p_choice=p_choice)
        g_chebysev = self._get_chebyshev_series(
            n=n,
            domain_min=0,
            domain_max=lambda_max,
            cut_off_frequency=cut_off_frequency,
            steep=steep,
        )

        print(g_chebysev(eigenvals))

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
    ) -> None:
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

        # get the unique eigenvalues
        L1 = self.sc.hodge_laplacian_matrix().toarray()
        U, eigenvals = get_eigendecomposition(lap_mat=L1)
        # get the true signal
        f_true = self.get_true_signal(f=flow, component=component)

        plt.figure(figsize=(15, 5))
        plt.scatter(eigenvals, U.T @ f_true)
        plt.scatter(eigenvals, f_cheb_tilde[-1])
        plt.title(
            "Frequency response on the eigenvalues vs chebyshev filter approx"
        )
        # add legend
        plt.legend(["True flow", "Chebyshev approx"])
