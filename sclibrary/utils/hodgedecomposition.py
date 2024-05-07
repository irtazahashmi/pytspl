import numpy as np
from scipy.sparse import csr_matrix


def get_divergence(
    incidence_matrix: np.ndarray, flow: np.ndarray
) -> np.ndarray:
    """
    Get the divergence of a flow on a graph.

    Args:
        incidence_matrix (np.ndarray): The incidence matrix of the graph (B1).
        flow (np.ndarray): The flow on the graph.
        round_sig_fig (int, optional): Round to significant figure.
        Defaults to 2.

    Returns:
        np.ndarray: The divergence of the flow.
    """
    divergence = csr_matrix(incidence_matrix, dtype=float).dot(flow)
    return divergence


def get_gradient_component(
    incidence_matrix: np.ndarray,
    flow: np.ndarray,
    round_fig: bool = True,
    round_sig_fig: int = 2,
) -> np.ndarray:
    """
    Calculate the gradient flow of a flow on a graph.

    Args:
        incidence_matrix (np.ndarray): The incidence matrix of the graph (B2).
        flow (np.ndarray): The flow on the graph.
        round_sig_fig (int, optional): Round to significant figure.
        Defaults to 2.

    Returns:
        np.ndarray: The gradient flow.
    """
    p = np.linalg.lstsq(incidence_matrix.T, flow, rcond=None)[0]
    gradient_flow = csr_matrix(incidence_matrix.T, dtype=float).dot(p)

    if round_fig:
        gradient_flow = np.round(gradient_flow, round_sig_fig)

    return gradient_flow


def get_curl_component(
    incidence_matrix: np.ndarray,
    flow: np.ndarray,
    round_fig: bool = True,
    round_sig_fig: int = 2,
) -> np.ndarray:
    """
    Calculate the curl flow of a flow on a graph.

    Args:
        incidence_matrix (np.ndarray): The incidence matrix of the graph (B2).
        flow (np.ndarray): The flow on the graph.
        round_sig_fig (int, optional): Round to significant figure.
        Defaults to 2.

    Returns:
        np.ndarray: The curl flow.
    """
    w = np.linalg.lstsq(incidence_matrix, flow, rcond=None)[0]
    curl_flow = csr_matrix(incidence_matrix, dtype=float).dot(w)

    if round_fig:
        curl_flow = np.round(curl_flow, round_sig_fig)

    return curl_flow


def get_harmonic_component(
    incidence_matrix_b1: np.ndarray,
    incidence_matrix_b2: np.ndarray,
    flow: np.ndarray,
    round_fig: bool = True,
    round_sig_fig: int = 2,
) -> np.ndarray:
    """
    Calculate the harmonic flow of a flow on a graph.

    Args:
        flow (np.ndarray): The flow on the graph.
        round_sig_fig (int, optional): Round to significant figure.
        Defaults to 2.

    Returns:
        np.ndarray: The harmonic flow.
    """
    gradient_flow = get_gradient_component(
        incidence_matrix=incidence_matrix_b1, flow=flow, round_fig=False
    )

    curl_flow = get_curl_component(
        incidence_matrix=incidence_matrix_b2, flow=flow, round_fig=False
    )

    harmonic_flow = flow - gradient_flow - curl_flow

    if round_fig:
        harmonic_flow = np.round(harmonic_flow, round_sig_fig)

    return harmonic_flow
