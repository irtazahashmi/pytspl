import numpy as np
import pytest

from sclibrary import SimplicialComplexNetwork


class TestSimplicialComplex:
    def test_shape(self, sc: SimplicialComplexNetwork):
        nodes, edges, triangles = 7, 10, 3
        assert sc.shape == (nodes, edges, triangles)

    def test_max_dim(self, sc: SimplicialComplexNetwork):
        assert sc.max_dim == 2

    def test_nodes(self, sc: SimplicialComplexNetwork):
        assert sc.nodes == {0, 1, 2, 3, 4, 5, 6}

    def test_edges(self, sc: SimplicialComplexNetwork):
        edges = sc.edges
        assert len(edges) == 10

    def test_is_connected(self, sc: SimplicialComplexNetwork):
        assert sc.is_connected

    def test_get_faces(self, sc: SimplicialComplexNetwork):
        simplex = [0, 1]
        faces = sc.get_faces(simplex=simplex)
        expected_faces = [(1,), (0,)]
        assert faces == expected_faces

    def test_get_cofaces(self, sc: SimplicialComplexNetwork):
        simplex = [0]
        cofaces = sc.get_cofaces(simplex=simplex)
        expected_cofaces = [(0, 2, 3), (0, 1, 2), (0, 3), (0, 1), (0, 2), (0,)]
        assert cofaces == expected_cofaces

    def test_identity_matrix(self, sc: SimplicialComplexNetwork):
        nodes = 7
        identity_matrix = sc.identity_matrix()
        assert identity_matrix.shape == (nodes, nodes)
        assert np.array_equal(identity_matrix, np.eye(nodes))

    def test_incidence_matrix(self, sc: SimplicialComplexNetwork):
        nodes, edges, triangles = 7, 10, 3
        inc_mat = sc.incidence_matrix(rank=1)
        assert inc_mat.shape == (nodes, edges)
        inc_mat_2 = sc.incidence_matrix(rank=2)
        assert inc_mat_2.shape == (edges, triangles)
        # Bk * Bk+1 = 0
        assert np.array_equal(
            inc_mat @ inc_mat_2, np.zeros((nodes, triangles))
        )

    def test_adjacency_matrix(self, sc: SimplicialComplexNetwork):
        nodes, num_edges = 7, 10
        adj_mat_0 = sc.adjacency_matrix(rank=0)
        adj_mat = sc.adjacency_matrix(rank=1)
        assert adj_mat_0.shape == (nodes, nodes)
        assert adj_mat.shape == (num_edges, num_edges)

    def test_laplacian_matrix(self, sc: SimplicialComplexNetwork):
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

    def test_lower_laplacian_matrix(self, sc: SimplicialComplexNetwork):
        # L(k, l) = Bk.T * Bk
        edges = 10
        lap_mat_1 = sc.lower_laplacian_matrix(rank=1)
        assert lap_mat_1.shape == (edges, edges)
        L_1 = sc.incidence_matrix(rank=1).T @ sc.incidence_matrix(rank=1)
        assert np.array_equal(lap_mat_1, L_1)

    def test_upper_laplacian_matrix(self, sc: SimplicialComplexNetwork):
        # L(k, l) = Bk+1 * Bk+1.T
        edges, k = 10, 1
        lap_mat_2 = sc.upper_laplacian_matrix(rank=k)
        assert lap_mat_2.shape == (edges, edges)
        L_2 = (
            sc.incidence_matrix(rank=k + 1) @ sc.incidence_matrix(rank=k + 1).T
        )
        assert np.array_equal(lap_mat_2, L_2)

    def test_hodge_laplacian_matrix(self, sc: SimplicialComplexNetwork):
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

    def test_hodge_laplacian_matrix_lower_upper(
        self, sc: SimplicialComplexNetwork
    ):
        # Lk = L(k, upper) + L(k, lower)
        k = 1
        L_1_calculated = sc.upper_laplacian_matrix(
            rank=k
        ) + sc.lower_laplacian_matrix(rank=k)
        assert np.array_equal(
            sc.hodge_laplacian_matrix(rank=k), L_1_calculated
        )

    def test_hodge_laplacian_matrix_is_orthogonal(
        self, sc: SimplicialComplexNetwork
    ):
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

    def test_apply_lower_shifting(self, sc: SimplicialComplexNetwork):
        flow = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        L_shifted = sc.apply_lower_shifting(flow=flow)
        assert L_shifted.shape == (10,)
        expected_flow = np.array([0, 0, 0, 0, 0, 1, -1, 2, 1, -1])
        assert np.array_equal(L_shifted, expected_flow)
        # apply shifting 2 times
        L_shifted = sc.apply_lower_shifting(flow=flow, steps=2)
        expected_flow = np.array([0, -1, 1, -1, 2, 5, -5, 8, 4, -4])
        assert np.array_equal(L_shifted, expected_flow)

    def test_apply_upper_shifting(self, sc: SimplicialComplexNetwork):
        flow = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        L_shifted = sc.apply_upper_shifting(flow=flow)
        assert L_shifted.shape == (10,)
        expected_flow = np.array([0, 0, 0, 0, 0, 0, 0, 1, -1, 1])
        assert np.array_equal(L_shifted, expected_flow)
        # apply shifting 2 times
        L_shifted = sc.apply_upper_shifting(flow=flow, steps=2)
        expected_flow = np.array([0, 0, 0, 0, 0, 0, 0, 3, -3, 3])
        assert np.array_equal(L_shifted, expected_flow)

    def test_apply_two_step_shifting(self, sc: SimplicialComplexNetwork):
        flow = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        L_shifted = sc.apply_two_step_shifting(flow=flow)
        assert L_shifted.shape == (10,)
        expected_flow = np.array([0, -1, 1, -1, 2, 5, -5, 11, 1, -1])
        assert np.array_equal(L_shifted, expected_flow)

    def test_get_simplicial_embeddings(self, sc: SimplicialComplexNetwork):
        flow = [0.03, 0.5, 2.38, 0.88, -0.53, -0.52, 1.08, 0.47, -1.17, 0.09]
        f_tilda_h, f_tilda_c, f_tilda_g = sc.get_simplicial_embeddings(
            flow=flow
        )
        exptected_h = np.array([-1.001])
        exptected_c = np.array([-1.0, -0.999, -0.997])
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
        assert np.allclose(np.round(f_tilda_h, 3), exptected_h)
        assert np.allclose(np.round(f_tilda_c, 3), exptected_c)
        assert np.allclose(np.round(f_tilda_g, 3), exptected_g)

    def test_eigedecomposition_error(self, sc: SimplicialComplexNetwork):
        component = "unknown"

        with pytest.raises(ValueError):
            sc.get_eigendecomposition(component=component)

    def test_get_component_coefficients(self, sc: SimplicialComplexNetwork):
        alpha_g = sc.get_component_coefficients(component="gradient")
        alpha_c = sc.get_component_coefficients(component="curl")
        alpha_h = sc.get_component_coefficients(component="harmonic")

        expected_g = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 1])
        expected_c = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
        expected_h = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        assert np.array_equal(alpha_g, expected_g)
        assert np.array_equal(alpha_c, expected_c)
        assert np.array_equal(alpha_h, expected_h)

    def test_get_component_coefficients_error(
        self, sc: SimplicialComplexNetwork
    ):
        component = "unknown"

        with pytest.raises(ValueError):
            sc.get_component_coefficients(component=component)
