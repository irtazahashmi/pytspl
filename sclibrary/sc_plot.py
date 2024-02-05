import itertools
from collections.abc import Iterable

import matplotlib.pyplot as plt
import networkx as nx


class SCPlot:
    def __init__(self, sc, pos: dict = None) -> None:
        self.sc = sc

        # plotting
        self.ax = self._setup_axes()
        self.pos = self._init_layout(pos)

    def _setup_axes(
        self,
    ) -> None:
        ax = plt.gca()
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.axis("equal")
        ax.axis("off")
        return ax

    def _init_layout(self, pos: dict) -> dict:
        layout = pos
        edges = self._get_edges()

        if pos is None:
            # Using spring layout
            G = nx.Graph()
            G.add_edges_from(edges)
            layout = nx.spring_layout(G)
            # set the axis limits to a square
            self.ax.set_xlim([-1.1, 1.1])
            self.ax.set_ylim([-1.1, 1.1])
        else:
            # set the axis limits to the bounding box of the nodes
            min_x = min([x[0] for x in pos.values()])
            max_x = max([x[0] for x in pos.values()])
            min_y = min([x[1] for x in pos.values()])
            max_y = max([x[1] for x in pos.values()])

            padding = (max_x - min_x) * 0.2

            self.ax.set_xlim([min_x - padding, max_x + padding])
            self.ax.set_ylim([min_y - padding, max_y + padding])

        return layout

    def _get_nodes(self):
        # generate 0-simplices
        return list(set(itertools.chain(*self.sc.simplices)))

    def _get_edges(self):
        # generate 1-simplices
        return list(
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
        node_size: int = 100,
        node_color: str = "#ff7f0e",
        edgecolor: str = "black",
        alpha: float = 0.8,
        margins=None,
        with_labels: bool = False,
    ):
        nodes = self._get_nodes()

        self.ax.scatter(
            [self.pos[node_id][0] for node_id in nodes],
            [self.pos[node_id][1] for node_id in nodes],
            s=node_size,
            c=node_color,
            edgecolor=edgecolor,
            alpha=alpha,
            zorder=2,
        )

        if margins is not None:
            if isinstance(margins, Iterable):
                self.ax.margins(*margins)
            else:
                self.ax.margins(margins)

        if with_labels:
            self.draw_node_labels()

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
        edge_width: float = 0.7,
        directed: bool = False,
        alpha: float = 0.8,
        arrowsize: int = 10,
    ):
        edges = self._get_edges()

        if directed:
            graph = nx.DiGraph()
            graph.add_edges_from(edges)
            nx.draw_networkx_edges(
                graph,
                pos=self.pos,
                edge_color=edge_color,
                width=edge_width,
                alpha=alpha,
                arrowsize=arrowsize,
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
                self.ax.add_line(line)

        # fill the 2-simplices (triangles)
        for i, j, k in self._get_triangles():
            (x0, y0) = self.pos[i]
            (x1, y1) = self.pos[j]
            (x2, y2) = self.pos[k]
            tri = plt.Polygon(
                [[x0, y0], [x1, y1], [x2, y2]],
                edgecolor="black",
                facecolor=plt.cm.Blues(0.6),
                alpha=0.4,
                lw=0.5,
                zorder=1,
            )
            self.ax.add_patch(tri)

    def draw_sc_network(
        self, directed: bool = False, with_labels: bool = False
    ) -> None:
        # draw the edges
        self.draw_sc_edges(directed=directed)
        # draw the nodes
        self.draw_sc_nodes()

        if with_labels:
            # draw the labels
            self.draw_node_labels()
