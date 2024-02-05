import numpy as np

from sclibrary.sc_plot import SCPlot
from toponetx.classes import SimplicialComplex

"""Module to analyze simplicial complex data."""


class SimplicialComplexNetwork:
    def __init__(self, simplices: list, pos: dict = None):
        """
        Creates a simplicial complex network from edge list.

        Args:
            simplices (list): List of simplices of the simplicial complex.
            pos (dict, optional): Dict of positions [node_id : (x, y)] is used for placing
            the 0-simplices. The standard nx spring layour is used otherwise.
            Defaults to None.
        """
        self.sc = SimplicialComplex(simplices=simplices)
        self.plot = SCPlot(sc=self, pos=pos)

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
    def simplices(self):
        return self.sc.simplices

    @property
    def is_connected(self) -> bool:
        """Returns True if the simplicial complex is connected, False otherwise."""
        return self.sc.is_connected()

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
