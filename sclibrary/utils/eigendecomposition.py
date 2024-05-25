import warnings

import numpy as np

"""Module to compute eigendecomposition"""


def get_harmonic_eigenvectors(hodgle_lap_mat: np.ndarray) -> tuple:
    """
    Calculate the harmonic eigenvectors of the Hodge Laplacian - e.g. L1.

    Args:
        hodgle_lap_mat (np.ndarray): The Hodge Laplacian matrix L(k)

    Returns:
        u_h (np.ndarray): The harmonic  eigenvectors U(H).
        eigenvalues (np.ndarray): The eigenvalues of the Hodge Laplacian.
    """
    eigenvectors, eigenvalues = get_eigendecomposition(hodgle_lap_mat)
    # get columns with zero eigenvalues
    u_h = eigenvectors[:, np.where(eigenvalues == 0)[0]]
    eigenvalues = eigenvalues[np.where(eigenvalues == 0)[0]]
    return u_h, eigenvalues


def get_curl_eigenvectors(upper_lap_mat: np.ndarray) -> tuple:
    """
    Calculate the curl eigenvectors of the upper Laplacian e.g. L1U.

    Args:
        upper_lap_mat (np.ndarray): The upper Laplacian matrix L(k, u).

    Returns:
        u_c (np.ndarray): The curl eigenvectors U(C)
        lambda_vals (np.ndarray): The eigenvalues of the upper Laplacian
    """
    eigenvectors, eigenvalues = get_eigendecomposition(upper_lap_mat)
    # get columns with non-zero eigenvalues
    u_c = eigenvectors[:, np.where(eigenvalues > 0)[0]]
    eigenvalues = eigenvalues[np.where(eigenvalues > 0)[0]]
    return u_c, eigenvalues


def get_gradient_eigenvectors(lower_lap_mat: np.ndarray) -> tuple:
    """
    Calculate the gradient eigenvectors of the lower Laplacian e.g. L1L.

    Args:
        lower_lap_mat (np.ndarray): The lower Laplacian matrix L(k, l)

    Returns:
        u_g (np.ndarray): The gradient eigenvectors U(G)
        lambda_vals (np.ndarray): The eigenvalues of the lower Laplacian
    """
    eigenvectors, eigenvalues = get_eigendecomposition(lower_lap_mat)
    # get columns with non-zero eigenvalues
    u_g = eigenvectors[:, np.where(eigenvalues > 0)[0]]
    eigenvalues = eigenvalues[np.where(eigenvalues > 0)[0]]
    return u_g, eigenvalues


def get_eigendecomposition(lap_mat: np.ndarray, tolerance=1e-6) -> tuple:
    """
    Calculate the eigenvectors of the Laplacian matrix using
    eigendecomposition.

    The eigendecomposition of the Hodge Laplacian is given by:
    L(k) = U(k) * lambda(k) * U(k).T

    Sorts the eigenvectors according to the sorted eigenvalues.

    Args:
        lap_mat (np.ndarray): The Laplacian matrix L(k).
        tolerance (float): The tolerance for eigenvalues to be considered zero.
        Defaults to 1e-6.

    Returns:
        eigenvectors (np.ndarray): The eigenvectors U(k)
        eigenvalues (np.ndarray): The eigenvalues.
    """
    assert isinstance(
        lap_mat, np.ndarray
    ), "Laplacian matrix must be a numpy array"

    eigenvalues, eigenvectors = np.linalg.eig(lap_mat)
    # set eigenvalues below tolerance to zero
    eigenvalues[eigenvalues < tolerance] = 0

    with warnings.catch_warnings(record=True):
        # sort the eigenvectors according to the sorted eigenvalues
        eigenvectors = eigenvectors[:, eigenvalues.argsort()].astype(float)
        # sort the eigenvalues
        eigenvalues = np.sort(eigenvalues).astype(float)

    return eigenvectors, eigenvalues
