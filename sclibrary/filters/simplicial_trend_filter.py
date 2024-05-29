"""Module for the simplicial trend filter.

We assume that the simplicial complex's underlying
signal is approximately either divergence-free or curl-free.

The regularized filter l1 and l2 are implemented.
"""

import cvxpy as cp
import numpy as np
from numpy.linalg import norm

from sclibrary.filters.base_filter import BaseFilter
from sclibrary.simplicial_complex.simplicial_complex import SimplicialComplex


class SimplicialTrendFilter(BaseFilter):

    def __init__(self, simplicial_complex: SimplicialComplex):
        """
        Initialize the simplicial trend filter using the
        simplicial complex.
        """
        super().__init__(simplicial_complex)

        self.history = {
            "filter": None,
            "frequency_responses": None,
            "component_flow": None,
            "errors": None,
        }

    def set_history(
        self,
        filter: np.ndarray,
        frequency_responses: np.ndarray,
        component_flow: np.ndarray,
        errors: np.ndarray,
    ) -> None:
        """
        Set the history of the filter.

        Args:
            filter (np.ndarray, None): The filter.
            frequency_responses (np.ndarray): _description_
            component_flow (np.ndarray): _description_
            errors (np.ndarray): _description_
        """
        self.history["filter"] = (
            filter.astype(float) if filter is not None else None
        )
        self.history["frequency_responses"] = frequency_responses.astype(float)
        self.history["component_flow"] = component_flow.astype(float)
        self.history["errors"] = errors.astype(float)

    def get_divergence_flow(self, f: np.ndarray) -> np.ndarray:
        """
        Get the divergence flow.

        Args:
            f (np.ndarray): The flow to compute the divergence.

        Returns:
            np.ndarray: The normalized divergence flow.
        """
        B1 = self.sc.incidence_matrix(rank=1)
        return norm(B1 @ f)

    def get_curl_flow(self, f: np.ndarray) -> np.ndarray:
        """
        Get the curl flow.

        Args:
            f (np.ndarray): The flow to compute the curl.

        ValueError:
            If the component is not 'divergence' or 'curl'.

        Returns:
            np.ndarray: The normalized curl flow.
        """
        B2 = self.sc.incidence_matrix(rank=2)
        return norm(B2.T @ f)

    def get_power_noise(
        self, flow: np.ndarray, component: str, snr_db: np.ndarray
    ) -> tuple:
        """
        Get the power noise and the signal to noise ratio.

        Args:
            flow (np.ndarray): The flow of the simplicial complex.
            component (str): The component to regularize. Choose between
            'divergence' and 'curl'.
            snr_db (np.ndarray): The signal to noise ratio in dB.

        Raises:
            ValueError: If the component is not 'divergence' or 'curl'.

        Returns:
            tuple: The power noise and the signal to noise ratio.
        """
        if component not in ["divergence", "curl"]:
            raise ValueError(
                "Invalid component. Choose between 'divergence' and 'curl'."
            )

        num_edges = len(self.sc.edges)
        # signal to noise ratio
        snr = 10 ** (snr_db / 10)
        # noise variance
        power_flow = norm(flow, 2)
        power_noise = power_flow / snr / num_edges
        return power_noise, snr

    def denoising_l2_regularizer(
        self,
        flow: np.ndarray,
        component: str,
        num_realizations: int,
        snr_db: np.ndarray,
        mu: float = 0.5,
    ) -> None:
        """
        Simplicial trend filter developed by regularizing the the total
        divergence and curl via their l2 norm for denoising.

        Args:
            flow (np.ndarray): The flow of the simplicial complex.
            component (str): The component to regularize. Choose between
            'divergence' and 'curl'.
            num_realizations (int): The number of realizations.
            snr_db (np.ndarray): The signal to noise ratio in dB.
            mu (float, optional): The regularization parameter.
            Defaults to 0.5.

        Raises:
            ValueError: If the component is not 'divergence' or 'curl'.
        """

        num_edges = len(self.sc.edges)
        power_noise, snr = self.get_power_noise(
            flow=flow, component=component, snr_db=snr_db
        )

        components = {
            "divergence": self.get_divergence_flow,
            "curl": self.get_curl_flow,
        }

        L1l = self.sc.lower_laplacian_matrix(rank=1)
        # identity matrix
        I = np.eye(num_edges)

        errors_noisy = np.zeros((len(snr), num_realizations))

        frequency_responses = np.zeros((num_edges, len(snr), num_realizations))
        errors = np.zeros((len(snr), num_realizations))
        component_flow = np.zeros((len(snr), num_realizations))

        for i in range(len(snr)):
            for j in range(num_realizations):

                # add noise to the flow
                random_noise = (
                    power_noise[i] * np.random.randn(num_edges, 1).flatten()
                )
                f_noisy = flow + random_noise

                # compute the error with noisy flow
                errors_noisy[i, j] = self.calculate_error_NRMSE(f_noisy, flow)

                # compute the filter
                H = np.linalg.inv(I + mu * L1l)
                frequency_responses[:, i, j] = H @ f_noisy
                # compute the error
                error = self.calculate_error_NRMSE(
                    frequency_responses[:, i, j], flow
                )

                # compute the divergence or curl
                comp_flow = components[component](
                    f=frequency_responses[:, i, j]
                )

                # store the results
                errors[i, j] = error
                component_flow[i, j] = comp_flow

            print(
                f"SNR: {snr[i]} dB - "
                + f"error noisy: {np.mean(errors_noisy[i:,])} - "
                + f"l2 error: {np.mean(errors[i, :])}"
            )

        errors_mean = np.mean(errors, axis=1)
        component_flow_mean = np.mean(component_flow, axis=1)

        self.set_history(
            filter=H,
            frequency_responses=frequency_responses,
            component_flow=component_flow_mean,
            errors=errors_mean,
        )

    @staticmethod
    def _solver(
        num_edges: int, f_noisy: np.ndarray, shift_operator: np.ndarray
    ) -> cp.Variable:
        f_opt = cp.Variable(num_edges)
        shifted_edge_flow = shift_operator @ f_opt
        objective = cp.Minimize(
            1 * cp.norm(f_noisy - f_opt, p=2)
            + 0.5 * cp.norm(shifted_edge_flow, p=1)
        )
        prob = cp.Problem(objective)
        prob.solve()
        return f_opt

    def denoising_l1_regularizer(
        self,
        flow: np.ndarray,
        shift_operator: np.ndarray,
        component: str,
        num_realizations: int,
        snr_db: np.ndarray,
    ):
        num_edges = len(self.sc.edges)
        power_noise, snr = self.get_power_noise(
            flow=flow, component=component, snr_db=snr_db
        )

        components = {
            "divergence": self.get_divergence_flow,
            "curl": self.get_curl_flow,
        }

        num_edges = len(self.sc.edges)
        # signal to noise ratio
        snr = 10 ** (snr_db / 10)
        # noise variance
        power_flow = norm(flow, 2)
        power_noise = power_flow / snr / num_edges

        frequency_responses = np.zeros((num_edges, len(snr), num_realizations))
        errors = np.zeros((len(snr), num_realizations))
        component_flow = np.zeros((len(snr), num_realizations))

        for i in range(len(snr)):
            for j in range(num_realizations):
                # add noise to the flow
                random_noise = (
                    power_noise[i] * np.random.randn(num_edges, 1).flatten()
                )
                f_noisy = flow + random_noise

                f_opt = self._solver(
                    num_edges=num_edges,
                    f_noisy=f_noisy,
                    shift_operator=shift_operator,
                )

                frequency_responses[:, i, j] = f_opt.value
                # compute the error
                error = self.calculate_error_NRMSE(
                    frequency_responses[:, i, j], flow
                )

                # compute the divergence or curl
                comp_flow = components[component](
                    f=frequency_responses[:, i, j]
                )

                # store the results
                errors[i, j] = error
                component_flow[i, j] = comp_flow

            print(f"SNR: {snr[i]} dB - l1 error: {np.mean(errors[i, :])}")

        errors_mean = np.mean(errors, axis=1)
        component_flow_mean = np.mean(component_flow, axis=1)

        # update the history
        self.set_history(
            filter=None,
            frequency_responses=frequency_responses,
            component_flow=component_flow_mean,
            errors=errors_mean,
        )
