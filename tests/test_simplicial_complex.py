import numpy as np
import pytest

from pytspl import SimplicialComplex


class TestSimplicialComplex:
    def test_shape(self, sc_mock: SimplicialComplex):
        nodes, edges, triangles = 7, 10, 3
        assert sc_mock.shape == (nodes, edges, triangles)

    def test_max_dim(self, sc_mock: SimplicialComplex):
        assert sc_mock.max_dim == 2

    def test_nodes(self, sc_mock: SimplicialComplex):
        assert sc_mock.nodes == [0, 1, 2, 3, 4, 5, 6]

    def test_edges(self, sc_mock: SimplicialComplex):
        edges = sc_mock.edges
        assert len(edges) == 10

    def test_simplices(self, sc_mock: SimplicialComplex):
        simplices = sc_mock.simplices
        expected_simplicies = [
            (0,),
            (1,),
            (2,),
            (3,),
            (4,),
            (5,),
            (6,),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (2, 3),
            (2, 5),
            (3, 4),
            (4, 5),
            (4, 6),
            (5, 6),
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
        ]
        assert simplices == expected_simplicies

    def test_get_faces(self, sc_mock: SimplicialComplex):
        simplex = [0, 1]
        faces = sc_mock.get_faces(simplex=simplex)
        expected_faces = [(0,), (1,)]
        assert faces == expected_faces

    def test_identity_matrix(self, sc_mock: SimplicialComplex):
        nodes = 7
        identity_matrix = sc_mock.identity_matrix()
        assert identity_matrix.shape == (nodes, nodes)
        assert np.array_equal(identity_matrix, np.eye(nodes))

    def test_incidence_matrix(self, sc_mock: SimplicialComplex):
        nodes, edges, triangles = 7, 10, 3
        inc_mat = sc_mock.incidence_matrix(rank=1).toarray()
        assert inc_mat.shape == (nodes, edges)
        inc_mat_2 = sc_mock.incidence_matrix(rank=2).toarray()
        assert inc_mat_2.shape == (edges, triangles)
        # Bk * Bk+1 = 0
        assert np.array_equal(
            inc_mat @ inc_mat_2, np.zeros((nodes, triangles))
        )

    def test_adjacency_matrix(self, sc_mock: SimplicialComplex):
        nodes = 7
        adj_mat = sc_mock.adjacency_matrix().toarray()
        assert adj_mat.shape == (nodes, nodes)

    def test_laplacian_matrix(self, sc_mock: SimplicialComplex):
        nodes = 7
        lap_mat = sc_mock.laplacian_matrix().toarray()
        assert lap_mat.shape == (nodes, nodes)
        # sum of each row is 0
        assert np.allclose(np.sum(lap_mat, axis=1), np.zeros(nodes))
        # sum of each column is 0
        assert np.allclose(np.sum(lap_mat, axis=0), np.zeros(nodes))
        # Laplacian matrix = B1@B1.T
        B1 = sc_mock.incidence_matrix(rank=1)
        expected = B1 @ B1.T
        assert np.array_equal(lap_mat, expected.toarray())

    def test_lower_laplacian_matrix(self, sc_mock: SimplicialComplex):
        # L(k, l) = Bk.T * Bk
        edges = 10
        lap_mat_1 = sc_mock.lower_laplacian_matrix(rank=1).toarray()
        assert lap_mat_1.shape == (edges, edges)
        B1 = sc_mock.incidence_matrix(rank=1)
        L1 = B1.T @ B1
        assert np.array_equal(lap_mat_1, L1.toarray())

    def test_upper_laplacian_matrix(self, sc_mock: SimplicialComplex):
        # L(k, l) = Bk+1 * Bk+1.T
        edges, k = 10, 1
        lap_mat_2 = sc_mock.upper_laplacian_matrix(rank=k).toarray()
        assert lap_mat_2.shape == (edges, edges)
        B2 = sc_mock.incidence_matrix(rank=k + 1)
        L2 = B2 @ B2.T
        assert np.array_equal(lap_mat_2, L2.toarray())

    def test_hodge_laplacian_matrix(self, sc_mock: SimplicialComplex):
        nodes, edges, k = 7, 10, 0
        # L0 = B1 * B1.T
        lap_mat_0 = sc_mock.hodge_laplacian_matrix(rank=k).toarray()
        assert lap_mat_0.shape == (nodes, nodes)
        B1 = sc_mock.incidence_matrix(rank=k + 1)
        L0 = B1 @ B1.T
        assert np.array_equal(lap_mat_0, L0.toarray())

        # Lk = Bk.T * Bk + Bk+1 * Bk+1.T for k > 0
        k = 1
        lap_mat_1 = sc_mock.hodge_laplacian_matrix(rank=k).toarray()
        assert lap_mat_1.shape == (edges, edges)
        B1 = sc_mock.incidence_matrix(rank=k)
        B2 = sc_mock.incidence_matrix(rank=k + 1)
        L1 = B1.T @ B1 + B2 @ B2.T

        assert np.array_equal(lap_mat_1, L1.toarray())

    def test_hodge_laplacian_matrix_lower_upper(
        self, sc_mock: SimplicialComplex
    ):
        # Lk = L(k, upper) + L(k, lower)
        k = 1
        L1_calculated = sc_mock.upper_laplacian_matrix(
            rank=k
        ) + sc_mock.lower_laplacian_matrix(rank=k)
        assert np.array_equal(
            sc_mock.hodge_laplacian_matrix(rank=k).toarray(),
            L1_calculated.toarray(),
        )

    def test_hodge_laplacian_matrix_is_orthogonal(
        self, sc_mock: SimplicialComplex
    ):
        edges, k = 10, 1
        L1 = sc_mock.hodge_laplacian_matrix(rank=k)
        L1u = sc_mock.upper_laplacian_matrix(rank=k)
        L1l = sc_mock.lower_laplacian_matrix(rank=k)
        # L(k) = L(k, upper) + L(k, lower)
        assert np.allclose(L1.toarray(), (L1u + L1l).toarray())
        # dot(L(k), L(k, upper), L(k, lower)) = 0
        assert np.array_equal(
            (L1 @ L1l @ L1u).toarray(), np.zeros((edges, edges))
        )

    def test_apply_lower_shifting(self, sc_mock: SimplicialComplex):
        flow = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        L_shifted = sc_mock.apply_lower_shifting(flow=flow)
        assert L_shifted.shape == (10,)
        expected_flow = np.array([0, 0, 0, 0, 0, 1, -1, 2, 1, -1])
        assert np.array_equal(L_shifted, expected_flow)
        # apply shifting 2 times
        L_shifted = sc_mock.apply_lower_shifting(flow=flow, steps=2)
        expected_flow = np.array([0, -1, 1, -1, 2, 5, -5, 8, 4, -4])
        assert np.array_equal(L_shifted, expected_flow)

    def test_apply_upper_shifting(self, sc_mock: SimplicialComplex):
        flow = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        L_shifted = sc_mock.apply_upper_shifting(flow=flow)
        assert L_shifted.shape == (10,)
        expected_flow = np.array([0, 0, 0, 0, 0, 0, 0, 1, -1, 1])
        assert np.array_equal(L_shifted, expected_flow)
        # apply shifting 2 times
        L_shifted = sc_mock.apply_upper_shifting(flow=flow, steps=2)
        expected_flow = np.array([0, 0, 0, 0, 0, 0, 0, 3, -3, 3])
        assert np.array_equal(L_shifted, expected_flow)

    def test_apply_two_step_shifting(self, sc_mock: SimplicialComplex):
        flow = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        L_shifted = sc_mock.apply_two_step_shifting(flow=flow)
        assert L_shifted.shape == (10,)
        expected_flow = np.array([0, -1, 1, -1, 2, 5, -5, 11, 1, -1])
        assert np.array_equal(L_shifted, expected_flow)

    def test_get_simplicial_embeddings(self, sc_mock: SimplicialComplex):
        flow = [0.03, 0.5, 2.38, 0.88, -0.53, -0.52, 1.08, 0.47, -1.17, 0.09]
        f_tilda_h, f_tilda_c, f_tilda_g = sc_mock.get_simplicial_embeddings(
            flow=flow
        )
        exptected_h = np.array([1.001])
        exptected_c = np.array([1.000, 0.999, 0.997])
        exptected_g = np.array(
            [
                1.0006,
                1.0013,
                1.0017,
                1.0029,
                0.9953,
                1.0041,
            ]
        )
        assert np.allclose(np.abs(f_tilda_h), exptected_h, atol=1e-3)
        assert np.allclose(np.abs(f_tilda_c), exptected_c, atol=1e-3)
        assert np.allclose(np.abs(f_tilda_g), exptected_g, atol=1e-3)

    def test_eigedecomposition_error(self, sc_mock: SimplicialComplex):
        component = "unknown"

        with pytest.raises(ValueError):
            sc_mock.get_eigendecomposition(component=component)

    def test_get_component_coefficients(self, sc_mock: SimplicialComplex):
        alpha_g = sc_mock.get_component_coefficients(component="gradient")
        alpha_c = sc_mock.get_component_coefficients(component="curl")
        alpha_h = sc_mock.get_component_coefficients(component="harmonic")

        expected_g = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 1])
        expected_c = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
        expected_h = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        assert np.array_equal(alpha_g, expected_g)
        assert np.array_equal(alpha_c, expected_c)
        assert np.array_equal(alpha_h, expected_h)

    def test_get_component_coefficients_error(
        self, sc_mock: SimplicialComplex
    ):
        component = "unknown"

        with pytest.raises(ValueError):
            sc_mock.get_component_coefficients(component=component)
