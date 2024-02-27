import itertools
from collections.abc import Iterable
from numbers import Number

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from sclibrary.simplicial_complex import SimplicialComplexNetwork


class SCPlot:
    def __init__(
        self,
        sc: SimplicialComplexNetwork,
        pos: dict = None,
    ) -> None:
        """
        Args:
            sc (SimplicialComplexNetwork): The simplicial complex network object.
            pos (dict, optional): Dict of positions [node_id : (x, y)] is used for placing
            the 0-simplices. The standard nx spring layour is used otherwise.
        """

        self.sc = sc
        self.pos = pos

    def _init_axes(self, ax) -> dict:
        layout = self.pos
        edges = self._get_edges()

        if self.pos is None:
            # Using spring layout
            G = nx.Graph()
            G.add_edges_from(edges)
            layout = nx.spring_layout(G)
            # set the axis limits to a square
            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
        else:
            # set the axis limits to the bounding box of the nodes
            min_x = min([x[0] for x in self.pos.values()])
            max_x = max([x[0] for x in self.pos.values()])
            min_y = min([x[1] for x in self.pos.values()])
            max_y = max([x[1] for x in self.pos.values()])

            padding = (max_x - min_x) * 0.2

            ax.set_xlim([min_x - padding, max_x + padding])
            ax.set_ylim([min_y - padding, max_y + padding])

        # layout configuration
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.axis("equal")
        ax.axis("off")

        return layout

    def _get_nodes(self):
        # generate 0-simplices
        return list(set(itertools.chain(*self.sc.simplices)))

    def _get_edges(self):
        # generate 1-simplices
        edges = list(
            set(
                itertools.chain(
                    *[
                        [
                            tuple(sorted((i, j)))
                            for i, j in itertools.combinations(simplex, 2)
                        ]
                        for simplex in self.sc.simplices
                    ]
                )
            )
        )
        # sort the edges
        edges = sorted(edges, key=lambda x: (x[0], x[1]))
        return edges

    def _get_triangles(self):
        # generate 2-simplices
        triangles = list(
            set(
                itertools.chain(
                    *[
                        [
                            tuple(sorted((i, j, k)))
                            for i, j, k in itertools.combinations(simplex, 3)
                        ]
                        for simplex in self.sc.simplices
                    ]
                )
            )
        )

        return triangles

    def draw_sc_nodes(
        self,
        node_size: int = 300,
        node_color: str = "#ff7f0e",
        node_edge_colors: str = "black",
        cmap=plt.cm.Blues,
        vmin=None,
        vmax=None,
        alpha: float = 0.8,
        margins=None,
        with_labels: bool = False,
        ax=None,
    ):

        if ax is None:
            ax = plt.gca()

        self._init_axes(ax=ax)

        if np.iterable(node_color) and np.alltrue(
            [isinstance(c, Number) for c in node_color]
        ):
            if cmap is not None:
                assert isinstance(cmap, mpl.colors.Colormap)
            else:
                cmap = plt.get_cmap()

            if vmin is None:
                # for more contrast
                vmin = min(node_color) - abs(min(node_color)) * 0.5
            if vmax is None:
                vmax = max(node_color)

            # add colorbar
            color_map = mpl.cm.ScalarMappable(cmap=cmap)
            color_map.set_clim(vmin=vmin, vmax=vmax)
            fig = ax.get_figure()
            fig.colorbar(mappable=color_map, ax=ax)

        nodes = self._get_nodes()

        node_collection = ax.scatter(
            [self.pos[node_id][0] for node_id in nodes],
            [self.pos[node_id][1] for node_id in nodes],
            s=node_size,
            c=node_color,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors=node_edge_colors,
            alpha=alpha,
        )

        if margins is not None:
            if isinstance(margins, Iterable):
                ax.margins(*margins)
            else:
                ax.margins(margins)

        if with_labels:
            self.draw_node_labels()

        node_collection.set_zorder(2)

    def draw_node_labels(
        self,
        font_size: float = 12,
        font_color: str = "k",
        font_weight: str = "normal",
        alpha=None,
    ) -> None:
        for node_id in self._get_nodes():
            (x, y) = self.pos[node_id]
            plt.text(
                x,
                y,
                node_id,
                fontsize=font_size,
                color=font_color,
                weight=font_weight,
                ha="center",
                va="center",
                alpha=alpha,
            )

    def draw_sc_edges(
        self,
        edge_color: str = "k",
        edge_width: float = 1.0,
        directed: bool = False,
        alpha: float = 0.8,
        arrowsize: int = 10,
        edge_cmap=plt.cm.Reds,
        edge_vmin=None,
        edge_vmax=None,
        ax=None,
    ):

        if ax is None:
            ax = plt.gca()

        fig = ax.get_figure()
        self._init_axes(ax=ax)

        if np.iterable(edge_color) and np.alltrue(
            [isinstance(c, Number) for c in edge_color]
        ):
            if edge_cmap is not None:
                assert isinstance(edge_cmap, mpl.colors.Colormap)
            else:
                edge_cmap = plt.get_cmap()

            if edge_vmin is None:
                # for more contrast
                edge_vmin = min(edge_color) - abs(min(edge_color)) * 0.5
            if edge_vmax is None:
                edge_vmax = max(edge_color)

            # add colorbar
            color_map = mpl.cm.ScalarMappable(
                cmap=edge_cmap,
            )

            color_map.set_clim(vmin=edge_vmin, vmax=edge_vmax)
            fig.colorbar(
                mappable=color_map,
                ax=ax,
            )

        edges = self._get_edges()

        if directed:
            graph = nx.DiGraph()
            graph.add_edges_from(edges)

            # reorder the edges to match the order of the edge colors
            edge_color = [
                edge_color[edges.index(edge)] for edge in list(graph.edges())
            ]

            nx.draw_networkx_edges(
                graph,
                pos=self.pos,
                edge_color=edge_color,
                width=edge_width,
                alpha=alpha,
                arrowsize=arrowsize,
                edge_cmap=edge_cmap,
                edge_vmin=edge_vmin,
                edge_vmax=edge_vmax,
                ax=ax,
            )

        else:
            for src_id, dest_id in edges:
                (x0, y0) = self.pos[src_id]
                (x1, y1) = self.pos[dest_id]
                line = plt.Line2D(
                    [x0, x1],
                    [y0, y1],
                    color=edge_color,
                    lw=edge_width,
                    alpha=alpha,
                    zorder=1,
                )
                ax.add_line(line)

        # fill the 2-simplices (triangles)
        for i, j, k in self._get_triangles():
            (x0, y0) = self.pos[i]
            (x1, y1) = self.pos[j]
            (x2, y2) = self.pos[k]
            tri = plt.Polygon(
                [[x0, y0], [x1, y1], [x2, y2]],
                edgecolor="k",
                facecolor=plt.cm.Blues(0.4),
                alpha=0.3,
                lw=0.5,
                zorder=0,
            )
            ax.add_patch(tri)

    def draw_edge_labels(
        self,
        edge_labels: dict,
        label_pos: float = 0.5,
        font_size: int = 10,
        font_color: str = "k",
        font_weight: str = "normal",
        alpha=None,
        ax=None,
    ):

        if ax is None:
            ax = plt.gca()

        self._init_axes(ax=ax)

        edge_items = {}
        for (src, dest), label in edge_labels.items():
            (x1, y1) = self.pos[src]
            (x2, y2) = self.pos[dest]
            (x, y) = (
                x1 * label_pos + x2 * (1.0 - label_pos),
                y1 * label_pos + y2 * (1.0 - label_pos),
            )

            t = ax.text(
                x,
                y,
                label,
                size=font_size,
                color=font_color,
                weight=font_weight,
                alpha=alpha,
                zorder=1,
            )

            edge_items[(src, dest)] = t

        return edge_items

    def draw_sc_network(
        self, directed: bool = False, with_labels: bool = False, ax=None
    ) -> None:

        if ax is None:
            ax = plt.gca()

        self._init_axes(ax=ax)

        # draw the edges
        self.draw_sc_edges(directed=directed, ax=ax)
        # draw the nodes
        self.draw_sc_nodes(ax=ax)

        if with_labels:
            # draw the labels
            self.draw_node_labels()

    def draw_flow(self, flow: np.ndarray, ax=None) -> None:
        if ax is None:
            ax = plt.gca()

        # self._init_axes(ax=ax)

        # get edge labels
        edges = self.sc.edges
        edge_labels = {}
        for i in range(len(edges)):
            edge_labels[edges[i][0], edges[i][1]] = flow[i]

        # plot nodes and edges
        self.draw_sc_nodes(ax=ax)
        self.draw_sc_edges(
            edge_color=list(edge_labels.values()),
            edge_width=10,
            directed=True,
            arrowsize=30,
            ax=ax,
        )

        # plot labels
        self.draw_node_labels(font_size=7)
        self.draw_edge_labels(edge_labels=edge_labels, font_size=15, ax=ax)

    def draw_hodge_decomposition(
        self,
        flow: np.ndarray,
        round_fig: bool = True,
        round_sig_fig: int = 2,
    ) -> None:
        fig = plt.figure(figsize=(15, 5))

        f_h, f_c, f_g = self.sc.get_hodgedecomposition(
            flow=flow, round_fig=round_fig, round_sig_fig=round_sig_fig
        )

        # gradient flow
        ax = fig.add_subplot(1, 3, 1)
        ax.set_title("f_g")
        self.draw_flow(flow=f_g, ax=ax)

        # curl flow
        ax = fig.add_subplot(1, 3, 2)
        ax.set_title("f_c")
        self.draw_flow(flow=f_c, ax=ax)

        # harmonic flow
        ax = fig.add_subplot(1, 3, 3)
        ax.set_title("f_h")
        self.draw_flow(flow=f_h, ax=ax)

        plt.show()

    def draw_gradient_eigenvectors(
        self,
        eigenvector_indices: np.ndarray = [],
        round_fig: bool = True,
        round_sig_fig: int = 2,
    ):

        u_g, eigenvals_g = self.sc.get_eigendecomposition(component="gradient")

        if len(eigenvector_indices) == 0:
            eigenvector_indices = range(len(eigenvals_g))

        # Assuming you have a total number of eigenvector_indices as num_plots
        num_plots = len(eigenvector_indices)
        # Calculate the number of columns needed
        num_cols = min(num_plots, 3)
        # Calculate the number of rows needed
        num_rows = num_plots // num_cols

        if num_plots % num_cols != 0:
            num_rows += 1

        Position = range(1, num_plots + 1)
        fig = plt.figure(1, figsize=(15, 5 * num_rows))

        for i, eig_vec in enumerate(eigenvector_indices):
            ax = fig.add_subplot(num_rows, num_cols, Position[i])
            ax.set_title(
                "λ_G = {}".format(round(eigenvals_g[eig_vec], round_sig_fig))
            )

            flow = u_g[:, eig_vec]

            if round_fig:
                flow = np.round(flow, round_sig_fig)

            self.draw_flow(flow=flow, ax=ax)

        plt.tight_layout()
        plt.show()

    def draw_curl_eigenvectors(
        self,
        eigenvector_indices: np.ndarray = [],
        round_fig: bool = True,
        round_sig_fig: int = 2,
    ):

        u_c, eigenvals_c = self.sc.get_eigendecomposition(component="curl")

        if len(eigenvector_indices) == 0:
            eigenvector_indices = range(len(eigenvals_c))

        # Assuming you have a total number of eigenvector_indices as num_plots
        num_plots = len(eigenvector_indices)
        # Calculate the number of columns needed
        num_cols = min(num_plots, 3)
        # Calculate the number of rows needed
        num_rows = num_plots // num_cols

        if num_plots % num_cols != 0:
            num_rows += 1

        Position = range(1, num_plots + 1)
        fig = plt.figure(1, figsize=(15, 5 * num_rows))

        for i, eig_vec in enumerate(eigenvector_indices):
            ax = fig.add_subplot(num_rows, num_cols, Position[i])
            ax.set_title(
                "λ_C = {}".format(round(eigenvals_c[eig_vec], round_sig_fig))
            )

            flow = u_c[:, eig_vec]

            if round_fig:
                flow = np.round(flow, round_sig_fig)

            self.draw_flow(flow=flow, ax=ax)

        plt.tight_layout()
        plt.show()

    def draw_harmonic_eigenvectors(
        self,
        eigenvector_indices: np.ndarray = [],
        round_fig: bool = True,
        round_sig_fig: int = 2,
    ):

        u_h, eigenvals_h = self.sc.get_eigendecomposition(component="harmonic")

        if len(eigenvector_indices) == 0:
            eigenvector_indices = range(len(eigenvals_h))

        # Assuming you have a total number of eigenvector_indices as num_plots
        num_plots = len(eigenvector_indices)
        # Calculate the number of columns needed
        num_cols = min(num_plots, 3)
        # Calculate the number of rows needed
        num_rows = num_plots // num_cols

        if num_plots % num_cols != 0:
            num_rows += 1

        Position = range(1, num_plots + 1)
        fig = plt.figure(1, figsize=(15, 5 * num_rows))

        for i, eig_vec in enumerate(eigenvector_indices):
            ax = fig.add_subplot(num_rows, num_cols, Position[i])
            ax.set_title(
                "λ_H = {}".format(round(eigenvals_h[eig_vec], round_sig_fig))
            )

            flow = u_h[:, eig_vec]

            if round_fig:
                flow = np.round(flow, round_sig_fig)

            self.draw_flow(flow=flow, ax=ax)

        plt.tight_layout()
        plt.show()
