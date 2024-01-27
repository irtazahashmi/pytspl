import numpy as np


def get_harmonic_eigenvectors(hodgle_lap_mat: np.ndarray) -> tuple:
    """
    Calculate the harmonic eigenvectors of the Hodge Laplacian.

    Args:
        hodgle_lap_mat (np.ndarray): The Hodge Laplacian matrix L(k)

    Returns:
        u_h (np.ndarray): The harmonic eigenvectors U(H).
        eigenvalues (np.ndarray): The eigenvalues of the Hodge Laplacian.
    """
    eigenvectors, eigenvalues = _get_eigendecomposition(hodgle_lap_mat)
    # get columns with zero eigenvalues
    u_h = eigenvectors[:, np.where(eigenvalues == 0)[0]]
    return u_h, eigenvalues


def get_curl_eigenvectors(upper_lap_mat: np.ndarray) -> tuple:
    """
    Calculate the curl eigenvectors of the upper Laplacian.

    Args:
        upper_lap_mat (np.ndarray): The upper Laplacian matrix L(k, u)

    Returns:
        u_c (np.ndarray): The curl eigenvectors U(C)
        lambda_vals (np.ndarray): The eigenvalues of the upper Laplacian
    """
    eigenvectors, eigenvalues = _get_eigendecomposition(upper_lap_mat)
    # get columns with non-zero eigenvalues
    u_c = eigenvectors[:, np.where(eigenvalues != 0)[0]]
    return u_c, eigenvalues


def get_gradient_eigenvectors(lower_lap_mat: np.ndarray) -> tuple:
    """
    Calculate the gradient eigenvectors of the lower Laplacian.

    Args:
        lower_lap_mat (np.ndarray): The lower Laplacian matrix L(k, l)

    Returns:
        u_g (np.ndarray): The gradient eigenvectors U(G)
        lambda_vals (np.ndarray): The eigenvalues of the lower Laplacian
    """
    eigenvectors, eigenvalues = _get_eigendecomposition(lower_lap_mat)
    # get columns with non-zero eigenvalues
    u_g = eigenvectors[:, np.where(eigenvalues != 0)[0]]
    return u_g, eigenvalues


def _get_eigendecomposition(lap_mat: np.ndarray, tolerance=1e-03) -> tuple:
    """
    Calculate the eigenvectors of the Laplacian matrix using eigendecomposition.

    The eigendecomposition of the Hodge Laplacian is given by:
    L(k) = U(k) * lambda(k) * U(k).T

    Args:
        lap_mat (np.ndarray): The Laplacian matrix L(k).
        tolerance (float): The tolerance for eigenvalues to be considered zero.

    Returns:
        eigenvectors (np.ndarray): The eigenvectors U(k)
        eigenvalues (np.ndarray): The eigenvalues.
    """
    eigenvalues, eigenvectors = np.linalg.eig(lap_mat)
    eigenvalues[eigenvalues < tolerance] = 0
    lambda_matrix = np.diag(eigenvalues)
    # verify L(k) = U(k) * lambda(k) * U(k).T
    assert np.allclose(
        np.rint(eigenvectors @ lambda_matrix @ eigenvectors.T),
        lap_mat,
    )
    return eigenvectors, eigenvalues