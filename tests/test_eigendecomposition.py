import numpy as np

from pytspl.decomposition.eigendecomposition import (
    get_curl_eigenpair,
    get_eigendecomposition,
    get_gradient_eigenpair,
    get_harmonic_eigenpair,
)
from pytspl.simplicial_complex import SimplicialComplex


class TestEigendecomoposition:
    def test_harmonic_eigenvectors(self, sc_mock: SimplicialComplex):
        k = 1
        L1 = sc_mock.hodge_laplacian_matrix(rank=k).toarray()
        u_h, _ = get_harmonic_eigenpair(L1)
        # u_h.T @ u_h = I
        assert np.allclose(u_h.T @ u_h, np.eye(u_h.shape[1]))

    def test_curl_eigenvectors(self, sc_mock: SimplicialComplex):
        k = 1
        L1u = sc_mock.upper_laplacian_matrix(rank=k).toarray()
        u_c, _ = get_curl_eigenpair(L1u)
        # u_c.T @ u_c = I
        assert np.allclose(u_c.T @ u_c, np.eye(u_c.shape[1]))

    def test_gradient_eigenvectors(self, sc_mock: SimplicialComplex):
        k = 1
        L1l = sc_mock.lower_laplacian_matrix(rank=k).toarray()
        u_g, _ = get_gradient_eigenpair(L1l)
        # u_g.T @ u_g = I
        assert np.allclose(u_g.T @ u_g, np.eye(u_g.shape[1]))

    def test_dimensions_add_up(self, sc_mock: SimplicialComplex):
        # Ng + Nc + Nh = N
        k = 1
        tolerance = 1e-6

        L1 = sc_mock.hodge_laplacian_matrix(rank=k).toarray()
        u_h, _ = get_harmonic_eigenpair(L1, tolerance)

        L1u = sc_mock.upper_laplacian_matrix(rank=k).toarray()
        u_c, _ = get_curl_eigenpair(L1u, tolerance)

        L1l = sc_mock.lower_laplacian_matrix(rank=k).toarray()
        u_g, _ = get_gradient_eigenpair(L1l, tolerance)

        assert u_h.shape[1] + u_c.shape[1] + u_g.shape[1] == L1.shape[0]

    def test_matrices_orthogonal(self, sc_mock: SimplicialComplex):
        k = 1
        tolerance = 1e-6

        L1 = sc_mock.hodge_laplacian_matrix(rank=k).toarray()
        u_h, _ = get_harmonic_eigenpair(L1, tolerance)

        L1u = sc_mock.upper_laplacian_matrix(rank=k).toarray()
        u_c, _ = get_curl_eigenpair(L1u, tolerance)

        L1l = sc_mock.lower_laplacian_matrix(rank=k).toarray()
        u_g, _ = get_gradient_eigenpair(L1l, tolerance)
        # U_h.T @ U_c = 0
        assert np.allclose(
            u_h.T @ u_c,
            np.zeros((u_h.shape[1], u_c.shape[1])),
        )
        # U_h.T @ U_g = 0
        assert np.allclose(
            u_h.T @ u_g,
            np.zeros((u_h.shape[1], u_g.shape[1])),
        )
        # U_c.T @ U_g = 0
        assert np.allclose(
            u_c.T @ u_g,
            np.zeros((u_c.shape[1], u_g.shape[1])),
        )

    def test_eigendecomposition(self, sc_mock: SimplicialComplex):
        k = 1
        L1 = sc_mock.hodge_laplacian_matrix(rank=k).toarray()

        eigenvectors, eigenvalues = get_eigendecomposition(L1)

        lambda_matrix = np.diag(eigenvalues)
        # verify L(k) = U(k) * lambda(k) * U(k).T
        assert np.allclose(
            eigenvectors @ lambda_matrix @ eigenvectors.T,
            L1,
        )
