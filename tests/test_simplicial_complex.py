import numpy as np
import pytest

from sclibrary import SimplicialComplex


class TestSimplicialComplex:
    def test_shape(self, sc: SimplicialComplex):
        nodes, edges, triangles = 7, 10, 3
        assert sc.shape == (nodes, edges, triangles)

    def test_max_dim(self, sc: SimplicialComplex):
        assert sc.max_dim == 2

    def test_nodes(self, sc: SimplicialComplex):
        assert sc.nodes == [0, 1, 2, 3, 4, 5, 6]

    def test_edges(self, sc: SimplicialComplex):
        edges = sc.edges
        assert len(edges) == 10

    def test_simplices(self, sc: SimplicialComplex):
        simplices = sc.simplices
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

    def test_get_faces(self, sc: SimplicialComplex):
        simplex = [0, 1]
        faces = sc.get_faces(simplex=simplex)
        expected_faces = [(0,), (1,)]
        assert faces == expected_faces

    def test_identity_matrix(self, sc: SimplicialComplex):
        nodes = 7
        identity_matrix = sc.identity_matrix()
        assert identity_matrix.shape == (nodes, nodes)
        assert np.array_equal(identity_matrix, np.eye(nodes))

    def test_incidence_matrix(self, sc: SimplicialComplex):
        nodes, edges, triangles = 7, 10, 3
        inc_mat = sc.incidence_matrix(rank=1)
        assert inc_mat.shape == (nodes, edges)
        inc_mat_2 = sc.incidence_matrix(rank=2)
        assert inc_mat_2.shape == (edges, triangles)
        # Bk * Bk+1 = 0
        assert np.array_equal(
            inc_mat @ inc_mat_2, np.zeros((nodes, triangles))
        )

    def test_adjacency_matrix(self, sc: SimplicialComplex):
        nodes = 7
        adj_mat = sc.adjacency_matrix()
        assert adj_mat.shape == (nodes, nodes)

    def test_laplacian_matrix(self, sc: SimplicialComplex):
        nodes = 7
        lap_mat = sc.laplacian_matrix()
        assert lap_mat.shape == (nodes, nodes)
        # sum of each row is 0
        assert np.allclose(np.sum(lap_mat, axis=1), np.zeros(nodes))
        # sum of each column is 0
        assert np.allclose(np.sum(lap_mat, axis=0), np.zeros(nodes))
        # Laplacian matrix = B1@B1.T
        expected = sc.incidence_matrix(rank=1) @ sc.incidence_matrix(rank=1).T
        assert np.array_equal(lap_mat, expected)

    def test_lower_laplacian_matrix(self, sc: SimplicialComplex):
        # L(k, l) = Bk.T * Bk
        edges = 10
        lap_mat_1 = sc.lower_laplacian_matrix(rank=1)
        assert lap_mat_1.shape == (edges, edges)
        L_1 = sc.incidence_matrix(rank=1).T @ sc.incidence_matrix(rank=1)
        assert np.array_equal(lap_mat_1, L_1)

    def test_upper_laplacian_matrix(self, sc: SimplicialComplex):
        # L(k, l) = Bk+1 * Bk+1.T
        edges, k = 10, 1
        lap_mat_2 = sc.upper_laplacian_matrix(rank=k)
        assert lap_mat_2.shape == (edges, edges)
        L_2 = (
            sc.incidence_matrix(rank=k + 1) @ sc.incidence_matrix(rank=k + 1).T
        )
        assert np.array_equal(lap_mat_2, L_2)

    def test_hodge_laplacian_matrix(self, sc: SimplicialComplex):
        nodes, edges, k = 7, 10, 0
        # L0 = B1 * B1.T
        lap_mat_0 = sc.hodge_laplacian_matrix(rank=k)
        assert lap_mat_0.shape == (nodes, nodes)
        L_0 = (
            sc.incidence_matrix(rank=k + 1) @ sc.incidence_matrix(rank=k + 1).T
        )
        assert np.array_equal(lap_mat_0, L_0)

        # Lk = Bk.T * Bk + Bk+1 * Bk+1.T for k > 0
        k = 1
        lap_mat_1 = sc.hodge_laplacian_matrix(rank=k)
        assert lap_mat_1.shape == (edges, edges)
        L_1 = (
            sc.incidence_matrix(rank=k).T @ sc.incidence_matrix(rank=k)
            + sc.incidence_matrix(rank=k + 1)
            @ sc.incidence_matrix(rank=k + 1).T
        )

        assert np.array_equal(lap_mat_1, L_1)

    def test_hodge_laplacian_matrix_lower_upper(self, sc: SimplicialComplex):
        # Lk = L(k, upper) + L(k, lower)
        k = 1
        L_1_calculated = sc.upper_laplacian_matrix(
            rank=k
        ) + sc.lower_laplacian_matrix(rank=k)
        assert np.array_equal(
            sc.hodge_laplacian_matrix(rank=k), L_1_calculated
        )

    def test_hodge_laplacian_matrix_is_orthogonal(self, sc: SimplicialComplex):
        edges, k = 10, 1
        l_1 = sc.hodge_laplacian_matrix(rank=k)
        l_1_u = sc.upper_laplacian_matrix(rank=k)
        l_1_l = sc.lower_laplacian_matrix(rank=k)
        # L(k) = L(k, upper) + L(k, lower)
        assert np.allclose(l_1, l_1_u + l_1_l)
        # dot(L(k), L(k, upper), L(k, lower)) = 0
        assert np.array_equal(
            np.dot(l_1, np.dot(l_1_u, l_1_l)), np.zeros((edges, edges))
        )

    def test_apply_lower_shifting(self, sc: SimplicialComplex):
        flow = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        L_shifted = sc.apply_lower_shifting(flow=flow)
        assert L_shifted.shape == (10,)
        expected_flow = np.array([0, 0, 0, 0, 0, 1, -1, 2, 1, -1])
        assert np.array_equal(L_shifted, expected_flow)
        # apply shifting 2 times
        L_shifted = sc.apply_lower_shifting(flow=flow, steps=2)
        expected_flow = np.array([0, -1, 1, -1, 2, 5, -5, 8, 4, -4])
        assert np.array_equal(L_shifted, expected_flow)

    def test_apply_upper_shifting(self, sc: SimplicialComplex):
        flow = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        L_shifted = sc.apply_upper_shifting(flow=flow)
        assert L_shifted.shape == (10,)
        expected_flow = np.array([0, 0, 0, 0, 0, 0, 0, 1, -1, 1])
        assert np.array_equal(L_shifted, expected_flow)
        # apply shifting 2 times
        L_shifted = sc.apply_upper_shifting(flow=flow, steps=2)
        expected_flow = np.array([0, 0, 0, 0, 0, 0, 0, 3, -3, 3])
        assert np.array_equal(L_shifted, expected_flow)

    def test_apply_two_step_shifting(self, sc: SimplicialComplex):
        flow = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        L_shifted = sc.apply_two_step_shifting(flow=flow)
        assert L_shifted.shape == (10,)
        expected_flow = np.array([0, -1, 1, -1, 2, 5, -5, 11, 1, -1])
        assert np.array_equal(L_shifted, expected_flow)

    def test_get_simplicial_embeddings(self, sc: SimplicialComplex):
        flow = [0.03, 0.5, 2.38, 0.88, -0.53, -0.52, 1.08, 0.47, -1.17, 0.09]
        f_tilda_h, __file__, f_tilda_g = sc.get_simplicial_embeddings(
            flow=flow
        )
        exptected_h = np.array([-1.001])
        exptected_g = np.array(
            [
                -1.001,
                -1.001,
                -1.002,
                -1.003,
                -0.995,
                1.004,
            ]
        )
        assert np.allclose(f_tilda_h, exptected_h, atol=1e-3)
        assert np.allclose(f_tilda_g, exptected_g, atol=1e-3)

    def test_eigedecomposition_error(self, sc: SimplicialComplex):
        component = "unknown"

        with pytest.raises(ValueError):
            sc.get_eigendecomposition(component=component)

    def test_get_component_coefficients(self, sc: SimplicialComplex):
        alpha_g = sc.get_component_coefficients(component="gradient")
        alpha_c = sc.get_component_coefficients(component="curl")
        alpha_h = sc.get_component_coefficients(component="harmonic")

        expected_g = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 1])
        expected_c = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
        expected_h = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        assert np.array_equal(alpha_g, expected_g)
        assert np.array_equal(alpha_c, expected_c)
        assert np.array_equal(alpha_h, expected_h)

    def test_get_component_coefficients_error(self, sc: SimplicialComplex):
        component = "unknown"

        with pytest.raises(ValueError):
            sc.get_component_coefficients(component=component)
