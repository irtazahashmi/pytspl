import numpy as np

from sclibrary.eigendecomposition import get_eigendecomposition
from sclibrary.freq_component import FrequencyComponent
from sclibrary.simplicial_complex import SimplicialComplexNetwork


class LSFilterDesign:

    def __init__(self, simplicial_complex: SimplicialComplexNetwork):
        self.sc = simplicial_complex

        self.f_estimated = None
        self.frequency_responses = None
        self.errors = None

    def _get_true_signal(self, component: str, f: np.ndarray) -> np.ndarray:
        """
        Get the true signal for the component.

        Args:
            component (str): The component to be extracted.
            f (np.ndarray): The signal to be filtered.

        Returns:
            np.ndarray: The true signal.
        """
        f_h, f_c, f_g = self.sc.get_hodgedecomposition(flow=f, round_fig=False)
        component_mapping = {
            FrequencyComponent.HARMONIC.value: f_h,
            FrequencyComponent.CURL.value: f_c,
            FrequencyComponent.GRADIENT.value: f_g,
        }

        try:
            f_true = component_mapping[component]
        except KeyError:
            raise ValueError(
                f"Invalid component {component}. Use 'harmonic', 'curl' or 'gradient'."
            )

        return f_true

    def _apply_filter(
        self,
        L: int,
        lap_matrix: np.ndarray,
        f: np.ndarray,
        f_true: np.ndarray,
        U1: np.ndarray,
        eigenvals: np.ndarray,
        alpha: np.ndarray,
    ) -> None:
        """
        Apply the filter to the signal using a type of the laplacian matrix.

        Args:
            L (int): The size of the filter.
            lap_matrix (np.ndarray): The laplacian matrix. It can be the Hodge Laplacian matrix,
            the upper or lower laplacian matrix.
            f (np.ndarray): The noisy signal to be filtered.
            f_true (np.ndarray): The true signal.
            U1 (np.ndarray): The eigenvectors of the laplacian matrix.
            eigenvals (np.ndarray): The eigenvalues of the laplacian matrix.
            alpha (np.ndarray): The coefficients of the filter.
        """

        # convert L1 to a numpy array with dtype object
        L1 = np.array(lap_matrix, dtype=object)

        filter_range = range(L)

        # create a matrix to store the system
        system_mat = np.zeros((len(eigenvals), len(filter_range)))

        # store the results
        errors = np.zeros((len(filter_range)))
        frequency_responses = np.zeros((len(U1), len(filter_range)))
        f_estimated = None

        for L in filter_range:

            system_mat[:, L] = np.power(eigenvals, L)

            # Least square solution to obtain the filter coefficients
            h = np.linalg.lstsq(system_mat, alpha, rcond=None)[0]

            # building the topological filter
            H_1 = np.zeros_like(L1, dtype=object)

            for l in range(len(h)):
                H_1 += h[l] * np.linalg.matrix_power(L1, l)

            # filter the signal
            f_estimated = H_1 @ f

            # compute the error for each filter size
            error = np.linalg.norm(f_estimated - f_true) / np.linalg.norm(
                f_true
            )
            errors[L] = error

            # filter frequency response
            H_1_tilda = np.diag(U1.T @ H_1 @ U1)
            frequency_responses[:, L] = H_1_tilda

        # update the results
        self.f_estimated = np.array(f_estimated).astype(float)
        self.frequency_responses = np.array(frequency_responses).astype(float)
        self.errors = np.array(errors).astype(float)

    # TODO: name this better
    def simplicial_filter(
        self, L: int, component: str, f: np.ndarray
    ) -> tuple:
        """
        LS based filter design for subcomponent extraction by filter H1.

        Hk = sum(l=0, L) h_l L^l

        In this case, we will use the Hodge Laplacian matrix L1 = L2 = L and α = β.

        Args:
            L (int): The size of the filter.
            component (str): The component to be extracted.
            f (np.ndarray): The signal to be filtered.

        Returns:
            tuple: The estimated signal, errors and frequency responses.
        """

        # eigendecomposition of the Hodge Laplacian matrix
        L1 = self.sc.hodge_laplacian_matrix(rank=1)
        U1, eigenvals = get_eigendecomposition(L1)

        # get the true signal
        f_true = self._get_true_signal(component, f)
        # get the component coefficients
        alpha = self.sc.get_component_coefficients(component=component)

        self._apply_filter(
            L=L,
            lap_matrix=L1,
            f=f,
            f_true=f_true,
            U1=U1,
            eigenvals=eigenvals,
            alpha=alpha,
        )

        # update private attributes for the plot
        self._f = f
        self._filter_range = range(L)
        self._component = f"{component} subcomponent"

    # TODO: name this better
    def subcomponent_extraction(
        self,
        L: int,
        component: str,
        f: np.ndarray,
        tolerance: float = 1e-6,
    ) -> None:
        """
        General LS based filter design for subcomponent extraction by filter H1.

        In this case:
        - The solution will have zero coefficients on the α for curl extraction (L1 = 0)
        - The solution will have zero coefficients on the β for gradient extraction (L2 = 0)

        Therefore, we will only consider the upper or lower part of the filter
        to do so.

        Args:
            L (int): The size of the filter.
            component (str): The component to be extracted.
            f (np.ndarray): The signal to be filtered.
            tolerance (float, optional): The tolerance to consider the eigenvalues as unique.
            Defaults to 1e-6.
        """

        # get the Laplacian matrix according to the component
        componenet_mapping = {
            FrequencyComponent.GRADIENT.value: self.sc.lower_laplacian_matrix(
                rank=1
            ),
            FrequencyComponent.CURL.value: self.sc.upper_laplacian_matrix(
                rank=1
            ),
        }
        lap_matrix = componenet_mapping[component]
        if lap_matrix is None:
            raise ValueError(
                f"Invalid component {component}. Use 'gradient' or 'curl'."
            )

        # get unique eigenvalues
        U1, eigenvals = get_eigendecomposition(lap_matrix, tolerance=tolerance)
        eigenvals = np.unique(eigenvals)
        # get the true signal
        f_true = self._get_true_signal(component=component, f=f)

        # filter coefficients
        # TODO: check if the coefficients are correct
        alpha = [0] + [1] * (len(eigenvals) - 1)

        self._apply_filter(
            L=L,
            lap_matrix=lap_matrix,
            f=f,
            f_true=f_true,
            U1=U1,
            eigenvals=eigenvals,
            alpha=alpha,
        )

    def general_filter(
        self,
        L1: int,
        L2: int,
        f: np.ndarray,
        tolerance: float = 1e-6,
    ) -> None:
        """
        Denoising by a general filter H1 with L1 != L2 = L and α != β.

        Args:
            L1 (int): The size of the filter for the gradient extraction.
            L2 (int): The size of the filter for the curl extraction.
            f (np.ndarray): The signal to be filtered.
            tolerance (float, optional): The tolerance to consider the eigenvalues as unique.
            Defaults to 1e-6.
        """

        f_est_g, f_est_c, f_est_h = 0, 0, 0

        # gradient extraction
        if L1 > 0:
            self.subcomponent_extraction(
                L=L1,
                component=FrequencyComponent.GRADIENT.value,
                f=f,
                tolerance=tolerance,
            )
            f_est_g = self.f_estimated

        # curl extraction
        if L2 > 0:
            self.subcomponent_extraction(
                L=L2,
                component=FrequencyComponent.CURL.value,
                f=f,
                tolerance=tolerance,
            )
            f_est_c = self.f_estimated

        # harmonic extraction
        f_est_h = f - f_est_g - f_est_c

        # estimated signal
        f_estimated = f_est_g + f_est_c + f_est_h

        # update the results
        self.f_estimated = f_estimated
        self.frequency_responses = None
        self.errors = None

        # update private attributes for the plot
        self._f = f
        self._component = "General filter"

    def calculate_error(self) -> float:
        """
        Calculate the error of the estimated signal.
        """
        return np.linalg.norm(self.f_estimated - self._f) / np.linalg.norm(
            self._f
        )

    def plot_errors(self, figsize: tuple = (15, 5)):
        """
        Plot the errors for each filter size.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize)
        plt.plot(self._filter_range, self.errors)
        plt.title(f"{self._component} error")
