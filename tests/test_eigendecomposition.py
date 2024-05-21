import numpy as np

from sclibrary import SimplicialComplex
from sclibrary.utils.eigendecomposition import (
    get_curl_eigenvectors,
    get_eigendecomposition,
    get_gradient_eigenvectors,
    get_harmonic_eigenvectors,
)


class TestEigendecomoposition:
    def test_harmonic_eigenvectors(self, sc: SimplicialComplex):
        k = 1
        l_1 = sc.hodge_laplacian_matrix(rank=k)
        u_h, _ = get_harmonic_eigenvectors(l_1)
        # u_h.T @ u_h = I
        assert np.allclose(u_h.T @ u_h, np.eye(u_h.shape[1]))

    def test_curl_eigenvectors(self, sc: SimplicialComplex):
        k = 1
        l_1_u = sc.upper_laplacian_matrix(rank=k)
        u_c, _ = get_curl_eigenvectors(l_1_u)
        # u_c.T @ u_c = I
        assert np.allclose(u_c.T @ u_c, np.eye(u_c.shape[1]))

    def test_gradient_eigenvectors(self, sc: SimplicialComplex):
        k = 1
        l_1_l = sc.lower_laplacian_matrix(rank=k)
        u_g, _ = get_gradient_eigenvectors(l_1_l)
        # u_g.T @ u_g = I
        assert np.allclose(
            np.round(u_g.T @ u_g, decimals=3), np.eye(u_g.shape[1])
        )

    def test_dimensions_add_up(self, sc: SimplicialComplex):
        # Ng + Nc + Nh = N
        k = 1
        l_1 = sc.hodge_laplacian_matrix(rank=k)
        u_h, _ = get_harmonic_eigenvectors(l_1)
        l_1_u = sc.upper_laplacian_matrix(rank=k)
        u_c, _ = get_curl_eigenvectors(l_1_u)
        l_1_l = sc.lower_laplacian_matrix(rank=k)
        u_g, _ = get_gradient_eigenvectors(l_1_l)
        assert u_h.shape[1] + u_c.shape[1] + u_g.shape[1] == l_1.shape[0]

    def test_matrices_orthogonal(self, sc: SimplicialComplex):
        k = 1
        l_1 = sc.hodge_laplacian_matrix(rank=k)
        u_h, _ = get_harmonic_eigenvectors(l_1)
        l_1_u = sc.upper_laplacian_matrix(rank=k)
        u_c, _ = get_curl_eigenvectors(l_1_u)
        l_1_l = sc.lower_laplacian_matrix(rank=k)
        u_g, _ = get_gradient_eigenvectors(l_1_l)
        # U_h.T @ U_c = 0
        assert np.allclose(
            np.round(u_h.T @ u_c, decimals=3),
            np.zeros((u_h.shape[1], u_c.shape[1])),
        )
        # U_h.T @ U_g = 0
        assert np.allclose(
            np.round(u_h.T @ u_g, decimals=3),
            np.zeros((u_h.shape[1], u_g.shape[1])),
        )
        # U_c.T @ U_g = 0
        assert np.allclose(
            np.round(u_c.T @ u_g, decimals=3),
            np.zeros((u_c.shape[1], u_g.shape[1])),
        )

    def test_eigendecomposition(self, sc: SimplicialComplex):
        k = 1
        l_1 = sc.hodge_laplacian_matrix(rank=k)

        tolerance = 1e-03
        eigenvectors, eigenvalues = get_eigendecomposition(
            l_1, tolerance=tolerance
        )
        lambda_matrix = np.diag(eigenvalues)
        # verify L(k) = U(k) * lambda(k) * U(k).T
        assert np.allclose(
            eigenvectors @ lambda_matrix @ eigenvectors.T,
            l_1,
            atol=tolerance,
        )
