from itertools import combinations
from typing import Hashable, Iterable

import numpy as np
from scipy.sparse import csr_matrix

from toponetx.classes import SimplicialComplex

"""Module to analyze simplicial complex data."""


class SimplicialComplexNetwork:
    def __init__(self, simplices: list):
        """
        Creates a simplicial complex network from edge list.

        Args:
            simplices (list): List of simplices of the simplicial complex.
            Defaults to None.
        """
        self.sc = SimplicialComplex(simplices=simplices)

    @property
    def shape(self) -> tuple:
        """Returns the shape of the simplicial complex."""
        return self.sc.shape

    @property
    def max_dim(self) -> int:
        """Returns the maximum dimension of the simplicial complex."""
        return self.sc.dim

    @property
    def nodes(self) -> set:
        """Returns the set of nodes in the simplicial complex."""
        return set(node for (node,) in self.sc.nodes)

    @property
    def edges(self) -> list[tuple]:
        """Returns the set of edges in the simplicial complex"""
        simplices = self.simplices
        edges = [simplex for simplex in simplices if len(simplex) == 2]
        edges = sorted(edges, key=lambda x: (x[0], x[1]))
        return edges

    @property
    def simplices(self) -> list[tuple]:
        simplices = set(simplex for simplex in self.sc.simplices)
        simplices = [tuple(simplex) for simplex in simplices]
        return simplices

    @property
    def is_connected(self) -> bool:
        """Returns True if the simplicial complex is connected, False otherwise."""
        return self.sc.is_connected()

    def get_faces(self, simplex: Iterable[Hashable]) -> set[tuple]:
        """
        Returns the faces of the simplex.

        Args:
            simplex (Iterable[Hashable]): Simplex for which to find the faces.
        """
        faceset = set()
        numnodes = len(simplex)
        for r in range(numnodes, 0, -1):
            for face in combinations(simplex, r):
                faceset.add(tuple(sorted(face)))
        k = len(simplex) - 1
        faceset = [face for face in faceset if len(face) == k]
        return faceset

    def get_cofaces(
        self, simplex: Iterable[Hashable], rank: int = 0
    ) -> list[tuple]:
        """
        Returns the cofaces of the simplex.

        Args:
            simplex (Iterable[Hashable]): Simplex for which to find the cofaces.
            rank (int): Rank of the cofaces. Defaults to 0. If rank is 0, returns
            all cofaces of the simplex.
        """
        cofaces = self.sc.get_cofaces(simplex=simplex, codimension=rank)
        cofaces = set(coface for coface in cofaces)
        cofaces = [tuple(coface) for coface in cofaces]
        return cofaces

    def identity_matrix(self) -> np.ndarray:
        """Identity matrix of the simplicial complex."""
        return np.eye(len(self.nodes))

    def incidence_matrix(self, rank: int) -> np.ndarray:
        """
        Computes the incidence matrix of the simplicial complex.

        Args:
            rank (int): Rank of the incidence matrix.

        Returns:
            np.ndarray: Incidence matrix of the simplicial complex.
        """
        inc_mat = self.sc.incidence_matrix(rank=rank).todense()
        return np.squeeze(np.asarray(inc_mat))

    def adjacency_matrix(self, rank: int) -> np.ndarray:
        """
        Computes the adjacency matrix of the simplicial complex.

        Args:
            rank (int): Rank of the adjacency matrix.

        Returns:
            np.ndarray: Adjacency matrix of the simplicial complex.
        """
        adj_mat = self.sc.adjacency_matrix(rank=rank).todense()
        return np.squeeze(np.asarray(adj_mat))

    def laplacian_matrix(self) -> np.ndarray:
        """
        Computes the Laplacian matrix of the simplicial complex.

        Returns:
            np.ndarray: Laplacian matrix of the simplicial complex.
        """
        lap_mat = self.sc.hodge_laplacian_matrix(rank=0).todense()
        return np.squeeze(np.asarray(lap_mat))

    def normalized_laplacian_matrix(self, rank: int) -> np.ndarray:
        """
        Computes the normalized Laplacian matrix of the simplicial complex.

        Args:
            rank (int): Rank of the normalized Laplacian matrix.

        Returns:
            np.ndarray: Normalized Laplacian matrix of the simplicial complex.
        """
        norm_lap_mat = self.sc.normalized_laplacian_matrix(rank=rank).todense()
        return np.squeeze(np.asarray(norm_lap_mat))

    def upper_laplacian_matrix(self, rank: int) -> np.ndarray:
        """
        Computes the upper Laplacian matrix of the simplicial complex.

        Args:
            rank (int): Rank of the upper Laplacian matrix.

        Returns:
            np.ndarray: Upper Laplacian matrix of the simplicial complex.
        """
        up_lap_mat = self.sc.up_laplacian_matrix(rank=rank).todense()
        return np.squeeze(np.asarray(up_lap_mat))

    def lower_laplacian_matrix(self, rank: int) -> np.ndarray:
        """
        Computes the lower Laplacian matrix of the simplicial complex.

        Args:
            rank (int): Rank of the lower Laplacian matrix.

        Returns:
            np.ndarray: Lower Laplacian matrix of the simplicial complex.
        """
        down_lap_mat = self.sc.down_laplacian_matrix(rank=rank).todense()
        return np.squeeze(np.asarray(down_lap_mat))

    def hodge_laplacian_matrix(self, rank: int) -> np.ndarray:
        """
        Computes the Hodge Laplacian matrix of the simplicial complex.

        Args:
            rank (int): Rank of the Hodge Laplacian matrix.

        Returns:
            np.ndarray: Hodge Laplacian matrix of the simplicial complex.
        """
        hodge_lap_mat = self.sc.hodge_laplacian_matrix(rank=rank).todense()
        return np.squeeze(np.asarray(hodge_lap_mat))

    def apply_lower_shifting(
        self, flow: np.ndarray, steps: int = 1
    ) -> np.ndarray:
        """
        Applies the lower shifting operator to the simplicial complex.

        Args:
            flow (np.ndarray): Flow on the simplicial complex.
            steps (int): Number of times to apply the lower shifting operator.
            Defaults to 1.

        Returns:
            np.ndarray: Lower shifted simplicial complex.
        """

        L1L = self.lower_laplacian_matrix(rank=1)

        if steps == 1:
            # L(1, l) @ f
            flow = csr_matrix(L1L).dot(flow)
        else:
            # L(1, l)**2 @ f
            flow = csr_matrix(L1L).dot(csr_matrix(L1L.T).dot(flow))

        return flow

    def apply_upper_shifting(
        self, flow: np.ndarray, steps: int = 1
    ) -> np.ndarray:
        """
        Applies the upper shifting operator to the simplicial complex.

        Args:
            flow (np.ndarray): Flow on the simplicial complex.
            steps (int): Number of times to apply the upper shifting operator.
            Defaults to 1.

        Returns:
            np.ndarray: Upper shifted simplicial complex.
        """

        L1U = self.upper_laplacian_matrix(rank=1)

        if steps == 1:
            # L(1, u) @ f
            flow = csr_matrix(L1U).dot(flow)
        else:
            # L(1, u)**2 @ f
            flow = csr_matrix(L1U).dot(csr_matrix(L1U.T).dot(flow))

        return flow

    def apply_two_step_shifting(self, flow: np.ndarray) -> np.ndarray:
        """
        Applies the two-step shifting operator to the simplicial complex.

        Args:
            flow (np.ndarray): Flow on the simplicial complex.

        Returns:
            np.ndarray: Two-step shifted simplicial complex.
        """
        two_step_lower_shifting = self.apply_lower_shifting(flow, steps=2)
        two_step_upper_shifting = self.apply_upper_shifting(flow, steps=2)
        return two_step_lower_shifting + two_step_upper_shifting
