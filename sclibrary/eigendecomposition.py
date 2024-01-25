import numpy as np


def get_eigendecomposition(hodgle_lap_mat, lower_lap_mat, upper_lap_mat):
    # harmonic eigenvectors
    u_h = _get_eigenvectors(hodgle_lap_mat)
    # gradient eigenvectors
    u_g = _get_eigenvectors(lower_lap_mat)
    ## curl eigenvectors
    u_c = _get_eigenvectors(upper_lap_mat)
    return u_h, u_g, u_c


def _get_eigenvectors(lap_mat):
    eigenvalues, eigenvectors = np.linalg.eig(lap_mat)
    lambda_values = np.diag(eigenvalues)
    lambda_values[lambda_values < 1e-3] = 0
    # L(k) = U(k) * lambda(k) * U(k).T
    assert np.allclose(
        np.rint(eigenvectors @ lambda_values @ np.linalg.inv(eigenvectors)),
        lap_mat,
    )
    return eigenvectors
