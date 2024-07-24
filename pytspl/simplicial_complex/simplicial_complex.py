"""Module for analyzing simplicial complex data."""

from itertools import combinations
from typing import Hashable, Iterable

import numpy as np
from scipy.sparse import csr_matrix

from pytspl.decomposition.eigendecomposition import (
    get_curl,
    get_curl_eigenpair,
    get_divergence,
    get_gradient_eigenpair,
    get_harmonic_eigenpair,
    get_total_variance,
)
from pytspl.decomposition.frequency_component import FrequencyComponent
from pytspl.decomposition.hodgedecomposition import (
    get_curl_flow,
    get_gradient_flow,
    get_harmonic_flow,
)


class SimplicialComplex:
    """Class for the simplicial complex network."""

    def __init__(
        self,
        nodes: list = [],
        edges: list = [],
        triangles: list = [],
        node_features: dict = {},
        edge_features: dict = {},
    ):
        """
        Create a simplicial complex network from edge list.

        Args:
            nodes (list, optional): List of nodes. Defaults to [].
            edges (list, optional): List of edges. Defaults to [].
            triangles (list, optional): List of triangles. Defaults to [].
            node_features (dict, optional): Dict of node features.
            Defaults to {}.
            edge_features (dict, optional): Dict of edge features.
            Defaults to {}.
        """
        self.nodes = nodes
        self.edges = edges
        self.triangles = triangles

        self.node_features = node_features
        self.edge_features = edge_features

        self.B1 = self.edges_to_B1(edges, len(nodes))
        self.B2 = self.triangles_to_B2(triangles, edges)

    def print_summary(self):
        """
        Print the summary of the simplicial complex.
        """
        print(f"Num. of nodes: {len(self.nodes)}")
        print(f"Num. of edges: {len(self.edges)}")
        print(f"Num. of triangles: {len(self.triangles)}")
        print(f"Shape: {self.shape}")
        print(f"Max Dimension: {self.max_dim}")

    def edges_to_B1(self, edges: list, num_nodes: int) -> np.ndarray:
        """
        Create the B1 matrix (node-edge) from the edges.

        Args:
            edges (list): List of edges.
            num_nodes (int): Number of nodes.

        Returns:
            np.ndarray: B1 matrix.
        """
        B1 = np.zeros((num_nodes, len(edges)))

        for j, edge in enumerate(edges):
            from_node, to_node = edge
            B1[from_node, j] = -1
            B1[to_node, j] = 1
        return B1

    def triangles_to_B2(self, triangles: list, edges: list) -> np.ndarray:
        """
        Create the B2 matrix (edge-triangle) from the triangles.

        Args:
            triangles (list): List of triangles.
            edges (list): List of edges.

        Returns:
            np.ndarray: B2 matrix.
        """
        B2 = np.zeros((len(edges), len(triangles)))
        for j, triangle in enumerate(triangles):
            a, b, c = triangle
            try:
                index_a = edges.index((a, b))
            except ValueError:
                index_a = edges.index((b, a))
            try:
                index_b = edges.index((b, c))
            except ValueError:
                index_b = edges.index((c, b))
            try:
                index_c = edges.index((a, c))
            except ValueError:
                index_c = edges.index((c, a))

            B2[index_a, j] = 1
            B2[index_c, j] = -1
            B2[index_b, j] = 1

        return B2

    def generate_coordinates(self) -> dict:
        """
        Generate the coordinates of the nodes using spring layout
        if the coordinates of the sc don't exist.

        Returns:
            dict: Coordinates of the nodes.
        """
        import networkx as nx

        print("WARNING: No coordinates found.")
        print("Generating coordinates using spring layout.")

        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)

        coordinates = nx.spring_layout(G)
        return coordinates

    @property
    def shape(self) -> tuple:
        """Return the shape of the simplicial complex."""
        return (len(self.nodes), len(self.edges), len(self.triangles))

    @property
    def max_dim(self) -> int:
        """Return the maximum dimension of the simplicial complex."""
        return max(len(simplex) for simplex in self.simplices) - 1

    @property
    def simplices(self) -> list[tuple]:
        """
        Get all the simplices of the simplicial complex.

        This includes 0-simplices (nodes), 1-simplices (edges), 2-simplices.

        Returns:
            list[tuple]: List of simplices.
        """
        nodes = [(node,) for node in self.nodes]
        return nodes + self.edges + self.triangles

    def edge_feature_names(self) -> list[str]:
        """Return the list of edge feature names."""
        if len(self.get_edge_features()) == 0:
            return []

        return list(list(self.get_edge_features().values())[0].keys())

    def get_node_features(self) -> list[dict]:
        """Return the list of node features."""
        return self.node_features

    def get_edge_features(self, name: str = None) -> list[dict]:
        """Return the list of edge features."""
        edge_features = self.edge_features
        if name:

            try:
                return {
                    key: value[name] for key, value in edge_features.items()
                }

            except KeyError:
                raise KeyError(
                    f"Edge feature {name} does not exist in"
                    + "the simplicial complex."
                )

        else:
            return edge_features

    def get_faces(self, simplex: Iterable[Hashable]) -> set[tuple]:
        """
        Return the faces of the simplex in order.

        Args:
            simplex (Iterable[Hashable]): Simplex for which to find the faces.

        Returns:
            set[tuple]: Set of faces of the simplex.
        """
        faceset = set()
        numnodes = len(simplex)
        for r in range(numnodes, 0, -1):
            for face in combinations(simplex, r):
                faceset.add(tuple(sorted(face)))
        k = len(simplex) - 1
        faceset = sorted([face for face in faceset if len(face) == k])
        return faceset

    def identity_matrix(self) -> np.ndarray:
        """Identity matrix of the simplicial complex."""
        return np.eye(len(self.nodes))

    def tocsr(self, matrix: np.ndarray) -> csr_matrix:
        """
        Convert a numpy array to a csr_matrix.

        Args:
            matrix (np.ndarray): Numpy array to convert.

        Returns:
            csr_matrix: Compressed Sparse Row matrix.
        """
        return csr_matrix(matrix, dtype=float)

    def incidence_matrix(self, rank: int) -> csr_matrix:
        """
        Compute the incidence matrix of the simplicial complex.

        Args:
            rank (int): Rank of the incidence matrix.

        Returns:
            csr_matrix: Incidence matrix of the simplicial complex.
        """
        if rank == 0:
            return np.ones(len(self.nodes), dtype=float)
        elif rank == 1:
            return self.tocsr(self.B1)
        elif rank == 2:
            return self.tocsr(self.B2)
        else:
            raise ValueError(
                "Rank cannot be larger than the dimension of the complex."
            )

    def adjacency_matrix(self) -> csr_matrix:
        """
        Compute the adjacency matrix of the simplicial complex.

        Returns:
            csr_matrix: Adjacency matrix of the simplicial complex.
        """
        adjacency_mat = np.zeros((self.B1.shape[0], self.B1.shape[0]))

        for col in range(self.B1.shape[1]):
            col_nozero = np.where(self.B1[:, col] != 0)[0]
            from_node, to_node = col_nozero[0], col_nozero[1]
            adjacency_mat[from_node, to_node] = 1
            adjacency_mat[to_node, from_node] = 1

        adjacency_mat = csr_matrix(adjacency_mat)
        return adjacency_mat

    def laplacian_matrix(self) -> csr_matrix:
        """
        Compute the Laplacian matrix of the simplicial complex.

        Returns:
            csr_matrix: Laplacian matrix of the simplicial complex.
        """
        B1 = self.incidence_matrix(rank=1)
        return B1 @ B1.T

    def lower_laplacian_matrix(self, rank: int = 1) -> csr_matrix:
        """
        Compute the lower Laplacian matrix of the simplicial complex.

        Args:
            rank (int): Rank of the lower Laplacian matrix.

        ValueError:
            If the rank is not 1 or 2.

        Returns:
            csr_matrix: Lower Laplacian matrix of the simplicial complex.
        """
        if rank == 1:
            B1 = self.incidence_matrix(rank=1)
            return B1.T @ B1
        elif rank == 2:
            B2 = self.incidence_matrix(rank=2)
            return B2.T @ B2
        else:
            raise ValueError("Rank must be either 1 or 2.")

    def upper_laplacian_matrix(self, rank: int = 1) -> csr_matrix:
        """
        Compute the upper Laplacian matrix of the simplicial complex.

        Args:
            rank (int): Rank of the upper Laplacian matrix.

        ValueError:
            If the rank is not 0 or 1.

        Returns:
            csr_matrix: Upper Laplacian matrix of the simplicial complex.
        """
        if rank == 0:
            return self.laplacian_matrix()
        elif rank == 1:
            B2 = self.incidence_matrix(rank=2)
            return B2 @ B2.T
        else:
            raise ValueError("Rank must be either 0 or 1.")

    def hodge_laplacian_matrix(self, rank: int = 1) -> csr_matrix:
        """
        Compute the Hodge Laplacian matrix of the simplicial complex.

        Args:
            rank (int): Rank of the Hodge Laplacian matrix.

        ValueError:
            If the rank is not 0, 1, or 2.

        Returns:
            csr_matrix: Hodge Laplacian matrix of the simplicial complex.
        """
        if rank == 0:
            return self.laplacian_matrix()
        elif rank == 1:
            return self.lower_laplacian_matrix(
                rank=rank
            ) + self.upper_laplacian_matrix(rank=rank)
        else:
            raise ValueError("Rank must be between 0 and 2.")

    def apply_lower_shifting(
        self, flow: np.ndarray, steps: int = 1
    ) -> np.ndarray:
        """
        Apply the lower shifting operator to the simplicial complex.

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
            flow = L1L @ flow
        else:
            # L(1, l)**2 @ f
            flow = L1L @ (L1L @ flow)

        return flow

    def apply_upper_shifting(
        self, flow: np.ndarray, steps: int = 1
    ) -> np.ndarray:
        """
        Apply the upper shifting operator to the simplicial complex.

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
            flow = L1U @ flow
        else:
            # L(1, u)**2 @ f
            flow = L1U @ (L1U @ flow)

        return flow

    def apply_k_step_shifting(
        self, flow: np.ndarray, steps: int = 2
    ) -> np.ndarray:
        """
        Apply the k-step shifting operator to the simplicial complex.

        Args:
            flow (np.ndarray): Flow on the simplicial complex.

        Returns:
            np.ndarray: k-step shifted simplicial complex.
        """
        two_step_lower_shifting = self.apply_lower_shifting(flow, steps=steps)
        two_step_upper_shifting = self.apply_upper_shifting(flow, steps=steps)
        return two_step_lower_shifting + two_step_upper_shifting

    def get_simplicial_embeddings(self, flow: np.ndarray) -> tuple:
        """
        Return the simplicial embeddings of the simplicial complex.

        Args:
            flow (np.ndarray): Flow on the simplicial complex.

        Returns:
            np.ndarray: Simplicial embeddings of the simplicial complex.
            Harmonic, curl, and gradient basis.
        """
        k = 1
        L1 = self.hodge_laplacian_matrix(rank=k).toarray()
        L1U = self.upper_laplacian_matrix(rank=k).toarray()
        L1L = self.lower_laplacian_matrix(rank=k).toarray()

        # eigendeomposition
        u_h, _ = get_harmonic_eigenpair(L1, tolerance=1e-3)
        u_c, _ = get_curl_eigenpair(L1U, 1e-3)
        u_g, _ = get_gradient_eigenpair(L1L, 1e-3)

        # each entry of an embedding represents the weight the flow has on the
        # corresponding eigenvector
        f_tilda_h = u_h.T @ flow
        f_tilda_c = u_c.T @ flow
        f_tilda_g = u_g.T @ flow

        return f_tilda_h, f_tilda_c, f_tilda_g

    def get_component_eigenpair(
        self,
        component: str = FrequencyComponent.HARMONIC.value,
        tolerance: float = 1e-3,
    ) -> tuple:
        """
        Return the eigendecomposition of the simplicial complex.

        Args:
            component (str, optional): Component of the eigendecomposition
            to return. Defaults to "harmonic".
            tolerance (float, optional): Tolerance for eigenvalues to be
            considered zero. Defaults to 1e-3.

        ValueError:
            If the component is not one of 'harmonic', 'curl', or 'gradient'.

        Returns:
            tuple: Eigenvectors and eigenvalues of the simplicial complex.
        """
        if component == FrequencyComponent.HARMONIC.value:
            L1 = self.hodge_laplacian_matrix(rank=1).toarray()
            u_h, eig_h = get_harmonic_eigenpair(L1, tolerance)
            return u_h, eig_h
        elif component == FrequencyComponent.CURL.value:
            L1U = self.upper_laplacian_matrix(rank=1).toarray()
            u_c, eig_c = get_curl_eigenpair(L1U, tolerance)
            return u_c, eig_c
        elif component == FrequencyComponent.GRADIENT.value:
            L1L = self.lower_laplacian_matrix(rank=1).toarray()
            u_g, eig_g = get_gradient_eigenpair(L1L, tolerance)
            return u_g, eig_g
        else:
            raise ValueError(
                "Invalid component. Choose from 'harmonic',"
                + "'curl', or 'gradient'."
            )

    def get_total_variance(self) -> np.ndarray:
        """
        Get the total variance of the SC.

        Returns:
            np.ndarray: The total variance of the SC.
        """
        laplacian_matrix = self.laplacian_matrix()
        return get_total_variance(laplacian_matrix)

    def get_divergence(self, flow: np.ndarray) -> np.ndarray:
        """
        Get the divergence of a flow on a graph.

        Args:
            flow (np.ndarray): The flow on the graph.

        Returns:
            np.ndarray: The divergence of the flow.
        """
        B1 = self.incidence_matrix(rank=1)
        return get_divergence(B1, flow)

    def get_curl(self, flow: np.ndarray) -> np.ndarray:
        """
        Get the curl of a flow on a graph.

        Args:
            flow (np.ndarray): The flow on the graph.

        Returns:
            np.ndarray: The curl of the flow.
        """
        B2 = self.incidence_matrix(rank=2)
        return get_curl(B2, flow)

    def get_component_flow(
        self,
        flow: np.ndarray,
        component: str = FrequencyComponent.GRADIENT.value,
        round_fig: bool = True,
        round_sig_fig: int = 2,
    ) -> np.ndarray:
        """
        Return the component flow of the simplicial complex
        using the hodgedecomposition.

        Args:
            flow (np.ndarray): Flow on the simplicial complex.
            component (str, optional): Component of the hodgedecomposition.
            Defaults to FrequencyComponent.GRADIENT.value.
            round_fig (bool, optional): Round the hodgedecomposition to the
            Default to True.
            round_sig_fig (int, optional): Round to significant figure.
            Defaults to 2.

        Returns:
            np.ndarray: Hodgedecomposition of the simplicial complex.
        """
        B1 = self.incidence_matrix(rank=1)
        B2 = self.incidence_matrix(rank=2)

        if component == FrequencyComponent.HARMONIC.value:
            f_h = get_harmonic_flow(
                B1=B1,
                B2=B2,
                flow=flow,
                round_fig=round_fig,
                round_sig_fig=round_sig_fig,
            )
            return f_h
        elif component == FrequencyComponent.CURL.value:
            f_c = get_curl_flow(
                B2=B2,
                flow=flow,
                round_fig=round_fig,
                round_sig_fig=round_sig_fig,
            )
            return f_c
        elif component == FrequencyComponent.GRADIENT.value:
            f_g = get_gradient_flow(
                B1=B1,
                flow=flow,
                round_fig=round_fig,
                round_sig_fig=round_sig_fig,
            )
            return f_g
        else:
            raise ValueError(
                "Invalid component. Choose from 'harmonic',"
                + "'curl', or 'gradient'."
            )
