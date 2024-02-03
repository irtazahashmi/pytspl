import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

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
        self.pos = pos

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

    def draw_2d(self, ax=None, node_radius: float = 0.02) -> None:
        """
        Draws a simplicial complex upto 2D from a list of simplices.

        Args:
            simplices (list):
                List of simplices to draw. Sub-simplices are not needed (only maximal).
                For example, the 2-simplex [1,2,3] will automatically generate the three
                1-simplices [1,2],[2,3],[1,3] and the three 0-simplices [1],[2],[3].
                When a higher order simplex is entered only its sub-simplices
                up to D=2 will be drawn.

            ax (matplotlib.pyplot.axes, optional): Defaults to None.
            node_radius (float, optional): Radius of the nodes. Defaults to 0.02.
        """
        # generate 0-simplices
        nodes = list(set(itertools.chain(*self.simplices)))

        # generate 1-simplices
        edges = list(
            set(
                itertools.chain(
                    *[
                        [
                            tuple(sorted((i, j)))
                            for i, j in itertools.combinations(simplex, 2)
                        ]
                        for simplex in self.simplices
                    ]
                )
            )
        )

        # generate 2-simplices
        triangles = list(
            set(
                itertools.chain(
                    *[
                        [
                            tuple(sorted((i, j, k)))
                            for i, j, k in itertools.combinations(simplex, 3)
                        ]
                        for simplex in self.simplices
                    ]
                )
            )
        )

        if ax is None:
            ax = plt.gca()

        if self.pos is None:
            # Using spring layout
            G = nx.Graph()
            G.add_edges_from(edges)
            self.pos = nx.spring_layout(G)
            # set it to a square
            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
        else:
            # radius depending on x, y
            node_radius = 0.02 * max(max_x - min_x, max_y - min_y) / 2

            # set it according to pos
            min_x = min([x[0] for x in self.pos.values()])
            max_x = max([x[0] for x in self.pos.values()])
            min_y = min([x[1] for x in self.pos.values()])
            max_y = max([x[1] for x in self.pos.values()])
            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.axis("off")

        # draw the edges
        for src_id, dest_id in edges:
            (x0, y0) = self.pos[src_id]
            (x1, y1) = self.pos[dest_id]
            line = plt.Line2D(
                [x0, x1], [y0, y1], color="black", zorder=1, lw=0.7
            )
            ax.add_line(line)

        # fill the triangles
        for i, j, k in triangles:
            (x0, y0) = self.pos[i]
            (x1, y1) = self.pos[j]
            (x2, y2) = self.pos[k]
            tri = plt.Polygon(
                [[x0, y0], [x1, y1], [x2, y2]],
                edgecolor="black",
                facecolor=plt.cm.Blues(0.6),
                zorder=2,
                alpha=0.4,
                lw=0.5,
            )
            ax.add_patch(tri)

        # draw the nodes
        for node_id in nodes:
            (x, y) = self.pos[node_id]
            circ = plt.Circle(
                [x, y],
                radius=node_radius,
                zorder=3,
                lw=3,
                edgecolor="Black",
                facecolor="#ff7f0e",
            )
            ax.add_patch(circ)
