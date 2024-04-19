from itertools import combinations
from typing import Hashable, Iterable

import numpy as np
from scipy.sparse import csr_matrix

from sclibrary.utils.eigendecomposition import (
    get_curl_eigenvectors,
    get_gradient_eigenvectors,
    get_harmonic_eigenvectors,
)
from sclibrary.utils.frequency_component import FrequencyComponent
from sclibrary.utils.hodgedecomposition import (
    get_curl_component,
    get_gradient_component,
    get_harmonic_component,
)
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
        """
        Returns True if the simplicial complex is connected, False
        otherwise.
        """
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
            simplex (Iterable[Hashable]): Simplex for which to find the
            cofaces.
            rank (int): Rank of the cofaces. Defaults to 0. If rank is 0,
            returns all cofaces of the simplex.
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

    def upper_laplacian_matrix(self, rank: int = 1) -> np.ndarray:
        """
        Computes the upper Laplacian matrix of the simplicial complex.

        Args:
            rank (int): Rank of the upper Laplacian matrix.

        Returns:
            np.ndarray: Upper Laplacian matrix of the simplicial complex.
        """
        up_lap_mat = self.sc.up_laplacian_matrix(rank=rank).todense()
        return np.squeeze(np.asarray(up_lap_mat))

    def lower_laplacian_matrix(self, rank: int = 1) -> np.ndarray:
        """
        Computes the lower Laplacian matrix of the simplicial complex.

        Args:
            rank (int): Rank of the lower Laplacian matrix.

        Returns:
            np.ndarray: Lower Laplacian matrix of the simplicial complex.
        """
        down_lap_mat = self.sc.down_laplacian_matrix(rank=rank).todense()
        return np.squeeze(np.asarray(down_lap_mat))

    def hodge_laplacian_matrix(self, rank: int = 1) -> np.ndarray:
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

    def get_simplicial_embeddings(self, flow: np.ndarray) -> tuple:
        """
        Returns the simplicial embeddings of the simplicial complex.

        Args:
            flow (np.ndarray): Flow on the simplicial complex.

        Returns:
            np.ndarray: Simplicial embeddings of the simplicial complex.
            Harmonic, curl, and gradient basis.
        """

        k = 1
        L1 = self.hodge_laplacian_matrix(rank=k)
        L1U = self.upper_laplacian_matrix(rank=k)
        L1L = self.lower_laplacian_matrix(rank=k)

        # eigendeomposition
        u_h, _ = get_harmonic_eigenvectors(L1)
        u_c, _ = get_curl_eigenvectors(L1U)
        u_g, _ = get_gradient_eigenvectors(L1L)

        # each entry of an embedding represents the weight the flow has on the
        # corresponding eigenvector
        # coefficients of the flow on the harmonic, curl, and gradient basis
        f_tilda_h = csr_matrix(u_h.T).dot(flow)
        f_tilda_c = csr_matrix(u_c.T).dot(flow)
        f_tilda_g = csr_matrix(u_g.T).dot(flow)

        return f_tilda_h, f_tilda_c, f_tilda_g

    def get_eigendecomposition(self, component: str = "harmonic") -> tuple:
        """
        Returns the eigendecomposition of the simplicial complex.

        Args:
            component (str, optional): Component of the eigendecomposition
            to return. Defaults to "harmonic".

        Returns:
            tuple: Eigenvectors and eigenvalues of the simplicial complex.
        """

        if component == FrequencyComponent.HARMONIC.value:
            L1 = self.hodge_laplacian_matrix(rank=1)
            u_h, eig_h = get_harmonic_eigenvectors(L1)
            return u_h, eig_h
        elif component == FrequencyComponent.CURL.value:
            L1U = self.upper_laplacian_matrix(rank=1)
            u_c, eig_c = get_curl_eigenvectors(L1U)
            return u_c, eig_c
        elif component == FrequencyComponent.GRADIENT.value:
            L1L = self.lower_laplacian_matrix(rank=1)
            u_g, eig_g = get_gradient_eigenvectors(L1L)
            return u_g, eig_g
        else:
            raise ValueError(
                "Invalid component. Choose from 'harmonic',"
                + "'curl', or 'gradient'."
            )

    def get_hodgedecomposition(
        self,
        flow: np.ndarray,
        round_fig: bool = True,
        round_sig_fig: int = 2,
    ) -> tuple:
        """
        Returns the hodgedecompositon of the simplicial complex.

        Args:
            flow (np.ndarray): Flow on the simplicial complex.
            round_fig (bool, optional): Round the hodgedecomposition to the
            round_sig_fig (int, optional): Round to significant figure.
            Defaults to 2.

        Returns:
            tuple: Harmonic, curl, and gradient components of the
            hodgedecomposition, respectively.
        """

        B1 = self.incidence_matrix(rank=1)
        B2 = self.incidence_matrix(rank=2)

        f_h = get_harmonic_component(
            incidence_matrix_b1=B1,
            incidence_matrix_b2=B2,
            flow=flow,
            round_fig=round_fig,
            round_sig_fig=round_sig_fig,
        )
        f_c = get_curl_component(
            incidence_matrix=B2,
            flow=flow,
            round_fig=round_fig,
            round_sig_fig=round_sig_fig,
        )
        f_g = get_gradient_component(
            incidence_matrix=B1,
            flow=flow,
            round_fig=round_fig,
            round_sig_fig=round_sig_fig,
        )

        return f_h, f_c, f_g

    def get_component_coefficients(
        self,
        component: str,
    ) -> np.ndarray:
        """
        Calculates the component coefficients of the given component using the
        order of the eigenvectors.

        Args:
            component (str): Component of the eigendecomposition to return.

        Raises:
            ValueError: If the component is not one of 'harmonic', 'curl',
            or 'gradient'.

        Returns:
            np.ndarray: The component coefficients of the simplicial complex
            for the given component.
        """

        L1 = self.hodge_laplacian_matrix(rank=1)

        U_H, e_h = self.get_eigendecomposition(
            FrequencyComponent.HARMONIC.value
        )
        U_C, e_c = self.get_eigendecomposition(FrequencyComponent.CURL.value)
        _, e_g = self.get_eigendecomposition(FrequencyComponent.GRADIENT.value)

        # concatenate the eigenvalues
        eigenvals = np.concatenate((e_h, e_c, e_g))

        # mask the eigenvectors
        mask = np.zeros(L1.shape[0])

        if component == FrequencyComponent.HARMONIC.value:
            mask[: U_H.shape[1]] = 1
        elif component == FrequencyComponent.CURL.value:
            mask[U_H.shape[1] : U_H.shape[1] + U_C.shape[1]] = 1
        elif component == FrequencyComponent.GRADIENT.value:
            mask[U_H.shape[1] + U_C.shape[1] :] = 1
        else:
            raise ValueError(
                "Invalid component. Choose from 'harmonic', 'curl', "
                + "or 'gradient'."
            )

        # sort mask according to eigenvalues
        mask = mask[np.argsort(eigenvals)]

        return mask

    def get_component_coefficients_by_type(self, component: str) -> np.ndarray:
        L1 = self.hodge_laplacian_matrix(rank=1)
        u_h, _ = self.get_eigendecomposition(FrequencyComponent.HARMONIC.value)
        u_g, _ = self.get_eigendecomposition(FrequencyComponent.GRADIENT.value)

        # mask the eigenvectors
        mask = np.zeros(L1.shape[0])

        if component == FrequencyComponent.HARMONIC.value:
            mask[: u_h.shape[1]] = 1
        elif component == FrequencyComponent.GRADIENT.value:
            mask[u_h.shape[1] : u_h.shape[1] + u_g.shape[1]] = 1
        elif component == FrequencyComponent.CURL.value:
            mask[u_h.shape[1] + u_g.shape[1] :] = 1

        else:
            raise ValueError(
                "Invalid component. Choose from 'harmonic', 'curl', "
                + "or 'gradient'."
            )

        return mask
