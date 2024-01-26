import numpy as np
import sympy


def get_harmonic_eigenvectors(hodgle_lap_mat: np.ndarray) -> tuple:
    """
    Calculate the harmonic eigenvectors of the Hodge Laplacian.

    Args:
        hodgle_lap_mat (np.ndarray): The Hodge Laplacian matrix L(k)

    Returns:
        u_h (np.ndarray): The harmonic eigenvectors U(H)
        lambda_vals (np.ndarray): The eigenvalues of the Hodge Laplacian
    """
    u_h, lambda_vals = _get_eigenvectors(hodgle_lap_mat)
    return u_h, lambda_vals


def get_curl_eigenvectors(upper_lap_mat: np.ndarray) -> tuple:
    """
    Calculate the curl eigenvectors of the upper Laplacian.

    Args:
        upper_lap_mat (np.ndarray): The upper Laplacian matrix L(k, u)

    Returns:
        u_c (np.ndarray): The curl eigenvectors U(C)
        lambda_vals (np.ndarray): The eigenvalues of the upper Laplacian
    """
    u_c, lambda_vals = _get_eigenvectors(upper_lap_mat)
    return u_c, lambda_vals


def get_gradient_eigenvectors(lower_lap_mat: np.ndarray) -> tuple:
    """
    Calculate the gradient eigenvectors of the lower Laplacian.

    Args:
        lower_lap_mat (np.ndarray): The lower Laplacian matrix L(k, l)

    Returns:
        u_g (np.ndarray): The gradient eigenvectors U(G)
        lambda_vals (np.ndarray): The eigenvalues of the lower Laplacian
    """
    u_g, lambda_vals = _get_eigenvectors(lower_lap_mat)
    return u_g, lambda_vals


def _get_eigenvectors(lap_mat: np.ndarray) -> tuple:
    """
    Calculate the eigenvectors of the Laplacian matrix using eigendecomposition.

    The eigendecomposition of the Hodge Laplacian is given by:
    L(k) = U(k) * lambda(k) * U(k).T

    Args:
        lap_mat (np.ndarray): The Laplacian matrix L(k),

    Returns:
        eigenvectors (np.ndarray): The eigenvectors U(k)
        eigenvalues (np.ndarray): The eigenvalues lambda(k)
    """
    eigenvalues, eigenvectors = np.linalg.eig(lap_mat)
    # remove small values due to numerical errors
    tolerance = 0.1 / np.abs(eigenvalues).max()
    eigenvectors[np.abs(eigenvectors) < tolerance] = 0
    lambda_values = np.diag(eigenvalues)
    lambda_values[lambda_values < 1e-3] = 0
    # L(k) = U(k) * lambda(k) * U(k).T
    assert np.allclose(
        np.rint(eigenvectors @ lambda_values @ np.linalg.inv(eigenvectors)),
        lap_mat,
    )
    return eigenvectors, lambda_values


def get_matrix_image(matrix: np.ndarray) -> tuple:
    """
    Calculate the image of a matrix.

    Args:
        matrix (np.ndarray): The matrix to calculate the image of.
    """
    matrix_t = sympy.Matrix(matrix.T)
    rref_matrix, pivot_columns = matrix_t.rref()
    return rref_matrix, pivot_columns
