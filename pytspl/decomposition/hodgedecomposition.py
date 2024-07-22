"""Module for calculating the Hodge decomposition of a flow on a graph.

The following components can be extracted:
- Total variance
- Divergence
- Curl
- Gradient component
- Curl component
- Harmonic component
"""

import numpy as np
from scipy.sparse import csr_matrix


def get_gradient_flow(
    B1: csr_matrix,
    flow: np.ndarray,
    round_fig: bool = True,
    round_sig_fig: int = 2,
) -> np.ndarray:
    """
    Calculate the gradient flow of a flow on a graph.

    Args:
        B1 (csr_matrix): The incidence matrix of the graph, nodes
        to edges (B1).
        flow (np.ndarray): The flow on the graph.
        round_sig_fig (int, optional): Round to significant figure.
        Defaults to 2.

    Returns:
        np.ndarray: The gradient flow.
    """
    p = np.linalg.lstsq(B1.T.toarray(), flow, rcond=None)[0]
    gradient_flow = B1.T @ p

    if round_fig:
        gradient_flow = np.round(gradient_flow, round_sig_fig)

    return gradient_flow


def get_curl_flow(
    B2: csr_matrix,
    flow: np.ndarray,
    round_fig: bool = True,
    round_sig_fig: int = 2,
) -> np.ndarray:
    """
    Calculate the curl flow of a flow on a graph.

    Args:
        B2 (csr_matrix): The incidence matrix of the graph, edges
        to triangles (B2).
        flow (np.ndarray): The flow on the graph.
        round_sig_fig (int, optional): Round to significant figure.
        Defaults to 2.

    Returns:
        np.ndarray: The curl flow.
    """
    w = np.linalg.lstsq(B2.toarray(), flow, rcond=None)[0]
    curl_flow = B2 @ w

    if round_fig:
        curl_flow = np.round(curl_flow, round_sig_fig)

    return curl_flow


def get_harmonic_flow(
    B1: csr_matrix,
    B2: csr_matrix,
    flow: np.ndarray,
    round_fig: bool = True,
    round_sig_fig: int = 2,
) -> np.ndarray:
    """
    Calculate the harmonic flow of a flow on a graph.

    Args:
        B1 (csr_matrix): The incidence matrix of the graph, nodes
        to edges (B1).
        B2 (csr_matrix): The incidence matrix of the graph,
        flow (np.ndarray): The flow on the graph.
        round_sig_fig (int, optional): Round to significant figure.
        Defaults to 2.

    Returns:
        np.ndarray: The harmonic flow.
    """
    gradient_flow = get_gradient_flow(B1=B1, flow=flow, round_fig=False)

    curl_flow = get_curl_flow(B2=B2, flow=flow, round_fig=False)

    harmonic_flow = flow - gradient_flow - curl_flow

    if round_fig:
        harmonic_flow = np.round(harmonic_flow, round_sig_fig)

    return harmonic_flow
