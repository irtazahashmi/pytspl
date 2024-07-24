"""Module to extract eigendecomposition into eigenvalues and eigenvectors.

After eigendecomposition, the components are extracted
into harmonic, curl and gradient eigenvalues and eigenvectors
(eigenpairs).
"""

import warnings

import numpy as np
from scipy.sparse.linalg import eigsh


def get_total_variance(laplacian_matrix: np.ndarray) -> float:
    """
    Calculate the total variance.

    Args:
        laplacian_matrix (np.ndarray): The Laplacian matrix L(k).

    Returns:
        float: The total variance.
    """
    eigenvecs, _ = get_eigendecomposition(laplacian_matrix)
    return np.diag(eigenvecs.T @ laplacian_matrix @ eigenvecs)


def get_divergence(B1: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Calculate the divergence.

    Args:
        B1 (np.ndarray): The incidence matrix B1.
        flow (np.ndarray): The edge flow.

    Returns:
        np.ndarray: The divergence of the flow.
    """
    return B1 @ flow


def get_curl(B2: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Calculate the curl.

    Args:
        B2 (np.ndarray): The incidence matrix B2.
        flow (np.ndarray): The edge flow.

    Returns:
        np.ndarray: The curl of the flow.
    """
    return B2.T @ flow


def get_harmonic_eigenpair(
    hodgle_lap_mat: np.ndarray, tolerance: float = np.finfo(float).eps
) -> tuple:
    """
    Calculate the harmonic eigenvectors of the Hodge Laplacian - e.g. L1
    with corresponding eigenvalues.

    Args:
        hodgle_lap_mat (np.ndarray): The Hodge Laplacian matrix L(k)
        tolerance (float): The tolerance for eigenvalues to be considered
        zero. Defaults to machine limits for floating point types.

    Returns:
        u_h (np.ndarray): The harmonic  eigenvectors U(H).
        eigenvalues (np.ndarray): The eigenvalues of the Hodge Laplacian.
    """
    eigenvectors, eigenvalues = get_eigendecomposition(hodgle_lap_mat)
    # get columns with zero eigenvalues as anything below tolerance
    # is considered zero
    u_h = eigenvectors[:, np.where(eigenvalues <= tolerance)[0]]
    eigenvalues = eigenvalues[np.where(eigenvalues <= tolerance)[0]]
    return u_h, eigenvalues


def get_curl_eigenpair(
    upper_lap_mat: np.ndarray, tolerance: float = np.finfo(float).eps
) -> tuple:
    """
    Calculate the curl eigenvectors of the upper Laplacian e.g. L1U
    with corresponding eigenvalues.

    Args:
        upper_lap_mat (np.ndarray): The upper Laplacian matrix L(k, u).
        tolerance (float): The tolerance for eigenvalues to be considered
        zero. Defaults to machine limits for floating point types.

    Returns:
        u_c (np.ndarray): The curl eigenvectors U(C)
        lambda_vals (np.ndarray): The eigenvalues of the upper Laplacian
    """
    eigenvectors, eigenvalues = get_eigendecomposition(upper_lap_mat)
    # get columns with non-zero eigenvalues
    u_c = eigenvectors[:, np.where(eigenvalues >= tolerance)[0]]
    eigenvalues = eigenvalues[np.where(eigenvalues >= tolerance)[0]]
    return u_c, eigenvalues


def get_gradient_eigenpair(
    lower_lap_mat: np.ndarray, tolerance: float = np.finfo(float).eps
) -> tuple:
    """
    Calculate the gradient eigenvectors of the lower Laplacian e.g. L1L
    with corresponding eigenvalues.

    Args:
        lower_lap_mat (np.ndarray): The lower Laplacian matrix L(k, l)
        tolerance (float): The tolerance for eigenvalues to be considered
        zero. Defaults to machine limits for floating point types.

    Returns:
        u_g (np.ndarray): The gradient eigenvectors U(G)
        lambda_vals (np.ndarray): The eigenvalues of the lower Laplacian
    """
    eigenvectors, eigenvalues = get_eigendecomposition(lower_lap_mat)
    # get columns with non-zero eigenvalues
    u_g = eigenvectors[:, np.where(eigenvalues > tolerance)[0]]
    eigenvalues = eigenvalues[np.where(eigenvalues > tolerance)[0]]
    return u_g, eigenvalues


def get_eigendecomposition(
    lap_mat: np.ndarray, tolerance: float = np.finfo(float).eps
) -> tuple:
    """
    Calculate the eigenvectors of the Laplacian matrix using
    eigendecomposition.

    The eigendecomposition of the Hodge Laplacian is given by:
    L(k) = U(k) * lambda(k) * U(k).T

    Sorts the eigenvectors according to the sorted eigenvalues.

    Args:
        lap_mat (np.ndarray): The Laplacian matrix L(k).
        tolerance (float): The tolerance for eigenvalues to be
        considered zero. Defaults to machine limits for floating point
        types.

    Returns:
        eigenvectors (np.ndarray): The eigenvectors U(k)
        eigenvalues (np.ndarray): The eigenvalues.
    """
    assert isinstance(
        lap_mat, np.ndarray
    ), "Laplacian matrix must be a numpy array"

    with warnings.catch_warnings(record=True):
        # eigenvalues, eigenvectors = np.linalg.eig(lap_mat)
        eigenvalues, eigenvectors = eigsh(lap_mat, k=lap_mat.shape[0])
        # set the values below tolerance to zero
        eigenvalues[np.abs(eigenvalues) < tolerance] = 0

    return eigenvectors, eigenvalues
