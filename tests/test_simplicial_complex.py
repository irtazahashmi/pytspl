import numpy as np
import pytest

from sclibrary.simplicial_complex import SimplicialComplexNetwork


@pytest.fixture
def edge_list():
    yield [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (3, 4),
        [1, 2, 3],
        [1, 3, 4],
    ]


@pytest.fixture
def sc(edge_list):
    yield SimplicialComplexNetwork(edge_list=edge_list)


class TestSimplicialComplex:
    def test_shape(self, sc: SimplicialComplexNetwork):
        nodes, edges, triangles = 5, 7, 2
        assert sc.shape == (nodes, edges, triangles)

    def test_max_dim(self, sc: SimplicialComplexNetwork):
        assert sc.max_dim == 2

    def test_nodes(self, sc: SimplicialComplexNetwork):
        assert sc.nodes == {0, 1, 2, 3, 4}

    def test_is_connected(self, sc: SimplicialComplexNetwork):
        assert sc.is_connected

    def test_identity_matrix(self, sc: SimplicialComplexNetwork):
        nodes = 5
        identity_matrix = sc.identity_matrix()
        assert identity_matrix.shape == (nodes, nodes)
        assert np.array_equal(identity_matrix, np.eye(5))

    def test_incidence_matrix(self, sc: SimplicialComplexNetwork):
        nodes, edges, triangles = 5, 7, 2
        inc_mat = sc.incidence_matrix(rank=1)
        assert inc_mat.shape == (nodes, edges)
        inc_mat_2 = sc.incidence_matrix(rank=2)
        assert inc_mat_2.shape == (edges, triangles)
        # Bk * Bk+1 = 0
        assert np.array_equal(inc_mat @ inc_mat_2, np.zeros((5, 2)))

    def test_adjacency_matrix(self, sc: SimplicialComplexNetwork):
        nodes, num_edges = 5, 7
        adj_mat_0 = sc.adjacency_matrix(rank=0)
        adj_mat = sc.adjacency_matrix(rank=1)
        assert adj_mat_0.shape == (nodes, nodes)
        assert adj_mat.shape == (num_edges, num_edges)

    def test_lower_laplacian_matrix(self, sc: SimplicialComplexNetwork):
        # L(k, l) = Bk.T * Bk
        nodes, edges, triangles = 5, 7, 2
        lap_mat_1 = sc.lower_laplacian_matrix(rank=1)
        assert lap_mat_1.shape == (edges, edges)
        L_1 = sc.incidence_matrix(rank=1).T @ sc.incidence_matrix(rank=1)
        assert np.array_equal(lap_mat_1, L_1)

    def test_upper_laplacian_matrix(self, sc: SimplicialComplexNetwork):
        # L(k, l) = Bk+1 * Bk+1.T
        edges, k = 7, 1
        lap_mat_2 = sc.upper_laplacian_matrix(rank=k)
        assert lap_mat_2.shape == (edges, edges)
        L_2 = (
            sc.incidence_matrix(rank=k + 1) @ sc.incidence_matrix(rank=k + 1).T
        )
        assert np.array_equal(lap_mat_2, L_2)

    def test_hodge_laplacian_matrix(self, sc: SimplicialComplexNetwork):
        nodes, edges, k = 5, 7, 0
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
