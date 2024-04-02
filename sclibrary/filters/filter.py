import numpy as np

from sclibrary.sc.simplicial_complex import SimplicialComplexNetwork
from sclibrary.utils.frequency_component import FrequencyComponent

"""Filter design base class."""


class Filter:

    def __init__(self, simplicial_complex: SimplicialComplexNetwork):
        self.sc = simplicial_complex

        self.history = {
            "filter": None,
            "f_estimated": None,
            "frequency_responses": None,
            "error_per_filter_size": None,
        }

    def _reset_history(self):
        """Reset the history of the filter design."""
        self.history = {
            "filter": None,
            "f_estimated": None,
            "frequency_responses": None,
            "error_per_filter_size": None,
        }

    def calculate_error(self, f_estimated: np.ndarray, f_true) -> float:
        """
        Calculate the error of the estimated signal.
        """
        return np.linalg.norm(f_estimated - f_true) / np.linalg.norm(f_true)

    def get_true_signal(self, component: str, f: np.ndarray) -> np.ndarray:
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

    def get_p_matrix(self, p_choice: str = "L1") -> np.ndarray:
        P_choices = {
            "L1": self.sc.hodge_laplacian_matrix(rank=1),
            "L1L": self.sc.lower_laplacian_matrix(rank=1),
            "L1U": self.sc.upper_laplacian_matrix(rank=1),
        }

        # eigenvalues
        try:
            P = P_choices[p_choice]
        except KeyError:
            raise ValueError(
                "Invalid P_choice. Choose from ['L1', 'L1L', 'L1U']"
            )

        return P
