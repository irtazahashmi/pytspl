"""Module for the simplicial trend filter.

We assume that the simplicial complex's underlying
signal is approximately either divergence-free or curl-free.

The regularized filter l1 and l2 are implemented.
"""

import cvxpy as cp
import numpy as np
from numpy.linalg import norm

from pytspl.filters.base_filter import BaseFilter
from pytspl.simplicial_complex.simplicial_complex import SimplicialComplex


class SimplicialTrendFilter(BaseFilter):
    """
    Simiplicial trend filter class for denoising and interpolation.
    Inherits from the filter base class.
    """

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
            "correlations": None,
        }

        self.components = {
            "divergence": self.sc.get_divergence,
            "curl": self.sc.get_curl,
        }

    @staticmethod
    def _corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute the correlation between two arrays."""
        corr = np.corrcoef(A, B)[0, 1]
        return corr

    @staticmethod
    def _solver(
        num_edges: int, f_noisy: np.ndarray, shift_operator: np.ndarray
    ) -> cp.Variable:
        """Least squares solver."""
        f_opt = cp.Variable(num_edges)
        shifted_edge_flow = shift_operator @ f_opt
        objective = cp.Minimize(
            1 * cp.norm(f_noisy - f_opt, p=2)
            + 0.5 * cp.norm(shifted_edge_flow, p=1)
        )
        prob = cp.Problem(objective)
        prob.solve()
        return f_opt

    def set_history(
        self,
        filter: np.ndarray,
        frequency_responses: np.ndarray,
        component_flow: np.ndarray,
        errors: np.ndarray,
        correlations: np.ndarray,
    ) -> None:
        """
        Set the history of the filter.

        Args:
            filter (np.ndarray, None): The filter.
            frequency_responses (np.ndarray): The frequency responses.
            component_flow (np.ndarray): The component flow.
            errors (np.ndarray): The errors.
            correlations (np.ndarray): The correlations.
        """
        self.history["filter"] = (
            filter.astype(float) if filter is not None else None
        )
        self.history["frequency_responses"] = frequency_responses.astype(float)
        self.history["component_flow"] = component_flow.astype(float)
        self.history["errors"] = errors.astype(float)
        self.history["correlations"] = correlations.astype(float)

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

        Returns:
            tuple: The power noise and the signal to noise ratio.
        """
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
        if component not in self.components.keys():
            raise ValueError(
                "Invalid component. Choose between 'divergence' and 'curl'."
            )

        num_edges = len(self.sc.edges)
        power_noise, snr = self.get_power_noise(
            flow=flow, component=component, snr_db=snr_db
        )

        L1l = self.sc.lower_laplacian_matrix(rank=1)
        # identity matrix
        I = np.eye(num_edges)

        errors_noisy = np.zeros((len(snr), num_realizations))

        frequency_responses = np.zeros((num_edges, len(snr), num_realizations))
        errors = np.zeros((len(snr), num_realizations))
        component_flow = np.zeros((len(snr), num_realizations))
        corrs = np.zeros((len(snr), num_realizations))

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
                errors[i, j] = self.calculate_error_NRMSE(
                    frequency_responses[:, i, j], flow
                )

                # compute the divergence or curl
                component_flow[i, j] = norm(
                    self.components[component](
                        flow=frequency_responses[:, i, j]
                    )
                )

                # compute the correlation
                corrs[i, j] = self._corr(flow, frequency_responses[:, i, j])

            print(
                f"SNR: {snr[i]} dB - "
                + f"error noisy: {np.mean(errors_noisy[i:,])} - "
                + f"l2 error: {np.mean(errors[i, :])}"
                + f" - corr: {np.mean(corrs[i, :])}"
            )

        errors_mean = np.mean(errors, axis=1)
        component_flow_mean = np.mean(component_flow, axis=1)
        corrs_mean = np.mean(corrs, axis=1)

        self.set_history(
            filter=H,
            frequency_responses=frequency_responses,
            component_flow=component_flow_mean,
            errors=errors_mean,
            correlations=corrs_mean,
        )

    def denoising_l1_regularizer(
        self,
        flow: np.ndarray,
        shift_operator: np.ndarray,
        component: str,
        num_realizations: int,
        snr_db: np.ndarray,
    ) -> None:
        """
        Simplicial trend filter developed by regularizing the the total
        divergence and curl via their l1 norm for denoising.

        Args:
            flow (np.ndarray): The flow of the simplicial complex.
            shift_operator (np.ndarray): The shift operator used in the
            optimization problem.
            component (str): The component to regularize. Choose between
            'divergence' and 'curl'.
            num_realizations (int): The number of realizations.
            snr_db (np.ndarray): The signal to noise ratio in dB.

        Raises:
            ValueError: If the component is not 'divergence' or 'curl'.
        """
        if component not in self.components.keys():
            raise ValueError(
                "Invalid component. Choose between 'divergence' and 'curl'."
            )

        num_edges = len(self.sc.edges)
        power_noise, snr = self.get_power_noise(
            flow=flow, component=component, snr_db=snr_db
        )

        num_edges = len(self.sc.edges)
        # signal to noise ratio
        snr = 10 ** (snr_db / 10)
        # noise variance
        power_flow = norm(flow, 2)
        power_noise = power_flow / snr / num_edges

        frequency_responses = np.zeros((num_edges, len(snr), num_realizations))
        errors = np.zeros((len(snr), num_realizations))
        component_flow = np.zeros((len(snr), num_realizations))
        correlations = np.zeros((len(snr), num_realizations))

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
                errors[i, j] = self.calculate_error_NRMSE(
                    frequency_responses[:, i, j], flow
                )

                # compute the divergence or curl
                component_flow[i, j] = norm(
                    self.components[component](
                        flow=frequency_responses[:, i, j]
                    )
                )

                # compute the correlation
                correlations[i, j] = self._corr(
                    flow, frequency_responses[:, i, j]
                )

            print(
                f"SNR: {snr[i]} dB - l1 error: {np.mean(errors[i, :])} - "
                + f"corr: {np.mean(correlations[i, :])}"
            )

        errors_mean = np.mean(errors, axis=1)
        component_flow_mean = np.mean(component_flow, axis=1)
        correlations_mean = np.mean(correlations, axis=1)

        # update the history
        self.set_history(
            filter=None,
            frequency_responses=frequency_responses,
            component_flow=component_flow_mean,
            errors=errors_mean,
            correlations=correlations_mean,
        )

    def interpolation_l1_regularizer(
        self,
        flow: np.ndarray,
        shift_operator: np.ndarray,
        component: str,
        ratio: int,
        num_realizations: int,
    ) -> None:
        """
        Simplicial trend filter developed by regularizing the the total
        divergence and curl via their l1 norm for interpolation.

        Args:
            flow (np.ndarray): The flow of the simplicial complex.
            shift_operator (np.ndarray): The shift operator used in the
            optimization problem.
            component (str): The component to regularize. Choose between
            'divergence' and 'curl'.
            ratio (int): The ratio of the number of nonzero nodes.
            num_realizations (int): The number of realizations.

        Raises:
            ValueError: If the component is not 'divergence' or 'curl'.
        """
        if component not in self.components.keys():
            raise ValueError(
                "Invalid component. Choose between 'divergence' and 'curl'."
            )

        num_edges = len(self.sc.edges)

        frequency_responses = np.zeros(
            (num_edges, len(ratio), num_realizations)
        )
        correlations = np.zeros((len(ratio), num_realizations))
        errors = np.zeros((len(ratio), num_realizations))
        component_flow = np.zeros((len(ratio), num_realizations))

        for i, r in enumerate(ratio):
            # the number of nonzero nodes
            M = np.floor(num_edges * r).astype(int)

            for j in range(num_realizations):

                mask = np.zeros(num_edges)
                mask[np.random.choice(num_edges, M, replace=False)] = 1
                sampling_matrix = np.diag(mask)
                # pick the rows that are not all zeros
                sampling_matrix = sampling_matrix[
                    np.any(sampling_matrix, axis=1), :
                ]

                # the labeled
                f_in = flow * mask

                # l1 norm based regularization
                # solve the optimization problem
                f_opt = cp.Variable(num_edges)
                shifted_edge_flow = shift_operator @ f_opt
                objective = cp.Minimize(
                    1 * cp.norm(f_opt - f_in, p=2)
                    + 0.5 * cp.norm(shifted_edge_flow, p=1)
                )
                constraints = [
                    sampling_matrix @ f_opt == sampling_matrix @ f_in
                ]
                prob = cp.Problem(objective, constraints)
                prob.solve()

                frequency_responses[:, i, j] = f_opt.value

                # compute the error
                errors[i, j] = self.calculate_error_NRMSE(
                    frequency_responses[:, i, j], flow
                )

                # compute the divergence or curl
                component_flow[i, j] = norm(
                    self.components[component](
                        flow=frequency_responses[:, i, j]
                    )
                )

                # compute the correlation
                correlations[i, j] = self._corr(
                    flow, frequency_responses[:, i, j]
                )

            print(
                f"Ratio: {r}  - error: {np.mean(errors[i, :])} - "
                + f"corr: {np.mean(correlations[i, :])}"
            )

        errors_mean = np.mean(errors, axis=1)
        component_flow_mean = np.mean(component_flow, axis=1)
        correlations_mean = np.mean(correlations, axis=1)

        # update the history
        self.set_history(
            filter=None,
            frequency_responses=frequency_responses,
            component_flow=component_flow_mean,
            errors=errors_mean,
            correlations=correlations_mean,
        )
