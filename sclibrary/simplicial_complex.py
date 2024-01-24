import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from toponetx.classes import SimplicialComplex

"""Module to analyze simplicial complex data."""


class SimplicialComplexNetwork:
    def __init__(self, edge_list: list, pos: dict = None):
        """
        Creates a simplicial complex network from edge list.

        Args:
            edge_list (list): List of edges. Each edge is a tuple of two nodes.
            pos (dict, optional): Dictionary of positions d:(x,y) is used for placing
            the 0-simplices. The standard nx spring layour is used otherwise.
            Defaults to None.
        """
        self.sc = SimplicialComplex(edge_list)
        self.pos = pos

    @property
    def shape(self) -> tuple:
        """Returns the shape of the simplicial complex."""
        return self.sc.shape

    @propertys
    def max_dim(self) -> int:
        """Returns the maximum dimension of the simplicial complex."""
        return self.sc.dim

    @property
    def nodes(self) -> set:
        return set(node for (node,) in self.sc.nodes)

    @property
    def simplices(self):
        return self.sc.simplices

    def is_connected(self) -> bool:
        return self.sc.is_connected()

    def identity_matrix(self) -> np.ndarray:
        return np.eye(len(self.nodes))

    def incidence_matrix(self, rank: int) -> np.ndarray:
        inc_mat = self.sc.incidence_matrix(rank=rank).todense()
        return np.squeeze(np.asarray(inc_mat))

    def adjacency_matrix(self, rank: int) -> np.ndarray:
        adj_mat = self.sc.adjacency_matrix(rank=rank).todense()
        return np.squeeze(np.asarray(adj_mat))

    def normalized_laplacian_matrix(self, rank: int) -> np.ndarray:
        norm_lap_mat = self.sc.normalized_laplacian_matrix(rank=rank).todense()
        return np.squeeze(np.asarray(norm_lap_mat))

    def upper_laplacian_matrix(self, rank: int) -> np.ndarray:
        up_lap_mat = self.sc.up_laplacian_matrix(rank=rank).todense()
        return np.squeeze(np.asarray(up_lap_mat))

    def lower_laplacian_matrix(self, rank: int) -> np.ndarray:
        down_lap_mat = self.sc.down_laplacian_matrix(rank=rank).todense()
        return np.squeeze(np.asarray(down_lap_mat))

    def hodge_laplacian_matrix(self, rank: int) -> np.ndarray:
        hodge_lap_mat = self.sc.hodge_laplacian_matrix(rank=rank).todense()
        return np.squeeze(np.asarray(hodge_lap_mat))

    def draw_2d(self, ax=None) -> None:
        """
        Draws a simplicial complex upto 2D from a list of simplices.

        Args:
            simplices (list[list[int]]):
                List of simplices to draw. Sub-simplices are not needed (only maximal).
                For example, the 2-simplex [1,2,3] will automatically generate the three
                1-simplices [1,2],[2,3],[1,3] and the three 0-simplices [1],[2],[3].
                When a higher order simplex is entered only its sub-simplices
                up to D=2 will be drawn.

            ax (matplotlib.pyplot.axes, optional): Defaults to None.
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
            # Creating a graph if pos is not given
            G = nx.Graph()
            G.add_edges_from(edges)
            self.pos = nx.spring_layout(G)
            # set it to a square
            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
        else:
            # set it according to pos
            ax.set_xlim(
                [
                    min([x[0] for x in self.pos.values()]),
                    max([x[0] for x in self.pos.values()]),
                ]
            )
            ax.set_ylim(
                [
                    min([x[1] for x in self.pos.values()]),
                    max([x[1] for x in self.pos.values()]),
                ]
            )

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.axis("off")

        # draw the edges
        for i, j in edges:
            (x0, y0) = self.pos[i]
            (x1, y1) = self.pos[j]
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
        for i in nodes:
            (x, y) = self.pos[i]
            circ = plt.Circle(
                [x, y],
                radius=0.02,
                zorder=3,
                lw=0.5,
                edgecolor="Black",
                facecolor="#ff7f0e",
            )
            ax.add_patch(circ)
