"""Module for plotting simplicial complexes."""

from collections.abc import Iterable
from numbers import Number

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from pytspl.decomposition.frequency_component import FrequencyComponent
from pytspl.simplicial_complex import SimplicialComplex


class SCPlot:
    """Class for plotting simplicial complexes."""

    def __init__(
        self,
        simplical_complex: SimplicialComplex,
        coordinates: dict = None,
    ) -> None:
        """
        Args:
            simplical_complex (SimplicialComplex): The simplicial
            complex network object.
            coordinates (dict, optional): Dict of positions
            [node_id : (x, y)] is used for placing the 0-simplices. The
            standard nx spring layer is used otherwise.
        """
        self.sc = simplical_complex
        self.pos = coordinates

    def _init_axes(self, ax) -> dict:
        """
        Initialize the axes for the plot. The axis limits are set to the
        bounding box of the nodes.

        Args:
            ax (matplotlib.axes.Axes): The axes object.

        Returns:
            dict: The layout of the nodes.
        """
        layout = self.pos

        if self.pos is None:
            # use spring layout if no coordinates are provided
            G = nx.Graph()
            G.add_edges_from(self.sc.edges)
            layout = nx.spring_layout(G)
            self.pos = layout
            # set the axis limits to a square
            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
        else:
            # scale the coordinates
            x = [x[0] for x in self.pos.values()]
            y = [x[1] for x in self.pos.values()]
            min_x, max_x = min(x), max(x)
            min_y, max_y = min(y), max(y)

            # add padding to the bounding box
            x_padding = (max_x - min_x) * 0.05
            y_padding = (max_y - min_y) * 0.05
            # set the axis limits according to the bounding box of the nodes
            ax.set_xlim([min_x - x_padding, max_x + x_padding])
            ax.set_ylim([min_y - y_padding, max_y + y_padding])

        # layout configuration
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.axis("off")

        return layout

    def create_edge_flow(self, flow: np.ndarray) -> dict:
        """
        Create a dictionary of edge flows from the flow array.

        Args:
            flow (np.ndarray): The flow on the edges.

        Returns:
            dict: The edge flow dictionary.
        """
        return dict(zip(self.sc.edges, flow))

    def draw_sc_nodes(
        self,
        node_size: int = 300,
        node_color: str = "#ff7f0e",
        node_edge_colors: str = "black",
        font_size: float = 12,
        font_color: str = "k",
        font_weight: str = "normal",
        cmap=plt.cm.Blues,
        vmin=None,
        vmax=None,
        alpha: float = 0.8,
        margins=None,
        with_labels: bool = False,
        ax=None,
    ) -> None:
        """
        Draw the nodes of the simplicial complex.

        Args:
            node_size (int, optional): The size of the nodes.
            Defaults to 300.
            node_color (str, optional): The color of the nodes.
            Defaults to '#ff7f0e'.
            node_edge_colors (str, optional): The color of the node edges.
            Defaults to 'black'.
            font_size (float, optional): The font size of the node labels.
            Defaults to 12.
            font_color (str, optional): The color of the node labels.
            Defaults to 'k'.
            font_weight (str, optional): The font weight of the node labels.
            Defaults to 'normal'.
            cmap (mpl.colors.Colormap, optional): The color map.
            Defaults to plt.cm.Blues.
            vmin (float, optional): The minimum value for the color map.
            Defaults to None.
            vmax (float, optional): The maximum value for the color map.
            Defaults to None.
            alpha (float, optional): The transparency of the nodes.
            Defaults to 0.8.
            margins (float, optional): The margins of the plot.
            Defaults to None.
            with_labels (bool, optional): Whether to show the node labels.
            Defaults to False.
            ax (matplotlib.axes.Axes, optional): The axes object.
            Defaults to None.
        """
        if ax is None:
            ax = plt.gca()

        self._init_axes(ax=ax)

        if np.iterable(node_color) and np.all(
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

        nodes = self.sc.nodes

        node_collection = ax.scatter(
            [self.pos[node_id][0] for node_id in nodes],
            [self.pos[node_id][1] for node_id in nodes],
            s=node_size,
            c=node_color,
            edgecolors=node_edge_colors,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
        )

        if margins is not None:
            if isinstance(margins, Iterable):
                ax.margins(*margins)
            else:
                ax.margins(margins)

        if with_labels:
            self._draw_node_labels(
                font_size=font_size,
                font_weight=font_weight,
                font_color=font_color,
                alpha=alpha,
            )

        node_collection.set_zorder(2)

    def _draw_node_labels(
        self,
        font_size: float = 12,
        font_color: str = "k",
        font_weight: str = "normal",
        alpha=None,
    ) -> None:
        """
        Draw the labels of the nodes.

        Args:
            font_size (float, optional): The font size of the node labels.
            Defaults to 12.
            font_color (str, optional): The color of the node labels.
            Defaults to 'k'.
            font_weight (str, optional): The font weight of the node labels.
            Defaults to 'normal'.
            alpha (float, optional): The transparency of the node labels.
            Defaults to None.
        """
        for node_id in self.sc.nodes:
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
        edge_flow: dict = None,
        edge_color: str = "lightblue",
        edge_width: float = 1.0,
        arrowsize: int = 10,
        edge_cmap=plt.cm.Blues,
        edge_vmin=None,
        edge_vmax=None,
        directed: bool = True,
        alpha: float = 0.8,
        ax=None,
    ) -> None:
        """
        Draw the edges of the simplicial complex.

        Args:
            edge_flow (dict, optional): The flow of the edges.
            e.g. {(0, 1): 0.5, (1, 2): 0.3, (2, 0): 0.2}.
            Defaults to None.
            edge_color (str, optional): The color of the edges.
            Defaults to 'lightblue'.
            edge_width (float, optional): The width of the edges.
            Defaults to 1.0.
            arrowsize (int, optional): The size of the arrows.
            Defaults to 10.
            edge_cmap (mpl.colors.Colormap, optional): The color map of
            the edges. Defaults to plt.cm.Blues.
            edge_vmin (float, optional): The minimum value for the color
            map. Defaults to None.
            edge_vmax (float, optional): The maximum value for the color
            map. Defaults to None.
            directed (bool, optional): Whether the edges are directed.
            Defaults to True.
            alpha (float, optional): The transparency of the edges.
            Defaults to 0.8.
            ax (matplotlib.axes.Axes, optional): The axes object.
            Defaults to None.
        """
        if edge_flow:
            assert isinstance(edge_flow, dict)

        if ax is None:
            ax = plt.gca()

        _, fig_height = ax.get_figure().get_size_inches()

        fig = ax.get_figure()
        self._init_axes(ax=ax)

        # if edge labels are provided, use them to color the edges
        if edge_flow is not None:
            edges = list(edge_flow.keys())
            edge_color = list(edge_flow.values())
        else:
            edges = self.sc.edges

        # create a graph
        graph = nx.DiGraph()
        graph.add_edges_from(edges)

        # edge color is iterable and all elements are numbers
        if np.iterable(edge_color) and np.all(
            [isinstance(c, Number) for c in edge_color]
        ):
            # check if edge_cmap is a colormap
            if edge_cmap is not None:
                assert isinstance(edge_cmap, mpl.colors.Colormap)
            else:
                edge_cmap = plt.get_cmap()

            # set the color map limits
            if edge_vmin is None:
                # for more contrast
                edge_vmin = min(edge_color) - abs(min(edge_color)) * 0.5
            if edge_vmax is None:
                edge_vmax = max(edge_color)

            # add colorbar
            color_map = mpl.cm.ScalarMappable(
                cmap=edge_cmap,
            )
            # set the color map limits
            color_map.set_clim(vmin=edge_vmin, vmax=edge_vmax)
            fig.colorbar(
                mappable=color_map,
                ax=ax,
            ).ax.tick_params(labelsize=fig_height * 2)

            # reorder the edges to match the order of the edge colors
            edge_color = [
                edge_color[edges.index(edge)] for edge in graph.edges()
            ]

        # draw edges
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
            arrows=directed,
        )

        # fill the 2-simplices (triangles)
        for i, j, k in self.sc.triangles:
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
    ) -> dict:
        """
        Draw the labels (flow) of the edges.

        Args:
            edge_labels (dict): The labels of the edges.
            e.g. {(0, 1): 0.5, (1, 2): 0.3, (2, 0): 0.2}
            Defaults to None.
            label_pos (float, optional): The position of the label.
            Defaults to 0.5.
            font_size (int, optional): The font size of the labels.
            Defaults to 10.
            font_color (str, optional): The color of the labels.
            Defaults to 'k'.
            font_weight (str, optional): The font weight of the labels.
            Defaults to 'normal'.
            alpha (float, optional): The transparency of the labels.
            Defaults to None.
            ax (matplotlib.axes.Axes, optional): The axes object.
            Defaults to None.
        """
        assert isinstance(edge_labels, dict)

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

    def draw_network(
        self,
        edge_flow=None,
        directed: bool = True,
        node_size: int = 400,
        edge_width: int = 5,
        arrowsize: int = 30,
        with_labels: bool = True,
        ax=None,
    ) -> None:
        """
        Draw the simplicial complex network with edge flow. If the flow
        is not provided, the network is drawn without flow.

        Args:
            edge_flow (dict, np.ndarray, list, optional): The labels of the
            edges. e.g. {(0, 1): 0.5, (1, 2): 0.3, (2, 0): 0.2}. You can also
            provide a numpy array of the flow. Defaults to None.
            directed (bool, optional): Whether the edges are directed.
            Defaults to True.
            node_size (int, optional): The size of the nodes.
            Defaults to 400.
            edge_width (int, optional): The width of the edges.
            Defaults to 5.
            arrowsize (int, optional): The size of the arrows.
            Defaults to 30.
            with_labels (bool, optional): Whether to show the node labels.
            Defaults to True.
            ax (matplotlib.axes.Axes, optional): The axes object.
            Defaults to None.
        """
        # initialize the axes
        if ax is None:
            ax = plt.gca()

        if isinstance(edge_flow, (np.ndarray, list)):
            edge_flow = self.create_edge_flow(flow=edge_flow)

        # if edge labels are provided, use them to color the edges
        if not np.iterable(edge_flow):
            edge_color = "lightblue"
        else:
            edge_color = list(edge_flow.values())

        # draw the nodes
        self.draw_sc_nodes(node_size=node_size, with_labels=with_labels, ax=ax)
        # draw the edges
        self.draw_sc_edges(
            edge_flow=edge_flow,
            edge_width=edge_width,
            arrowsize=arrowsize,
            directed=directed,
            ax=ax,
        )

        # plot edge labels
        if with_labels and np.all([isinstance(c, Number) for c in edge_color]):
            self.draw_edge_labels(edge_labels=edge_flow, font_size=15, ax=ax)

    def draw_hodge_decomposition(
        self,
        flow: np.ndarray,
        component=None,
        round_fig: bool = True,
        round_sig_fig: int = 2,
        figsize=(15, 5),
    ) -> None:
        """
        Draw the Hodge decomposition of the flow.

        Args:
            flow (np.ndarray): The flow on the edges.
            component (str, optional): The component of the flow to draw.
            If None, all three components are drawn. Defaults to None.
            round_fig (bool, optional): Whether to round the figures.
            Defaults to True.
            round_sig_fig (int, optional): The number of significant figures
            to round to. Defaults to 2.
            figsize (tuple, optional): The size of the figure.
            Defaults to (15, 5).

        Raises:
            ValueError: If an invalid component is provided.
        """
        fig = plt.figure(figsize=figsize)

        if component is not None:
            component_flow = self.sc.get_component_flow(
                flow=flow,
                component=component,
                round_fig=round_fig,
                round_sig_fig=round_sig_fig,
            )
            # create a single figure
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(f"f_{component}")
            self.draw_network(edge_flow=component_flow, ax=ax)

        # if no component is specified, draw all three components
        else:

            f_g = self.sc.get_component_flow(
                flow=flow,
                component=FrequencyComponent.GRADIENT.value,
                round_fig=round_fig,
                round_sig_fig=round_sig_fig,
            )

            f_c = self.sc.get_component_flow(
                flow=flow,
                component=FrequencyComponent.CURL.value,
                round_fig=round_fig,
                round_sig_fig=round_sig_fig,
            )

            f_h = self.sc.get_component_flow(
                flow=flow,
                component=FrequencyComponent.HARMONIC.value,
                round_fig=round_fig,
                round_sig_fig=round_sig_fig,
            )

            # gradient flow
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title("f_g")
            self.draw_network(edge_flow=f_g, ax=ax1)

            # curl flow
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title("f_c")
            self.draw_network(edge_flow=f_c, ax=ax2)

            # harmonic flow
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title("f_h")
            self.draw_network(edge_flow=f_h, ax=ax3)

        plt.show()

    def draw_eigenvectors(
        self,
        component: str,
        eigenvector_indices: np.ndarray = [],
        round_fig: bool = True,
        round_sig_fig: int = 2,
        with_labels: bool = True,
        figsize=(15, 5),
    ):
        """
        Draw the eigenvectors for the given component and eigenvalue
        indices using eigendecomposition.

        Args:
            component (str): The component of the eigenvectors to draw.
            eigenvector_indices (np.ndarray, optional): The indices of
            the eigenvectors to draw. Defaults to [].
            round_fig (bool, optional): Whether to round the figures.
            Defaults to True.
            round_sig_fig (int, optional): The number of significant
            figures to round to. Defaults to 2.
            with_labels (bool, optional): Whether to show the node labels.
            Defaults to True.
            figsize (tuple, optional): The size of the figure. Defaults to
            (15, 5).
        """
        viz_per_row = 3

        U, eigenvals = self.sc.get_component_eigenpair(component=component)

        # if no eigenvector indices are provided, draw all eigenvectors
        if len(eigenvector_indices) == 0:
            eigenvector_indices = range(len(eigenvals))

        # Assuming you have a total number of eigenvector_indices as num_plots
        num_plots = len(eigenvector_indices)
        # Calculate the number of columns needed
        num_cols = min(num_plots, viz_per_row)
        # Calculate the number of rows needed
        num_rows = num_plots // num_cols

        if num_plots % num_cols != 0:
            num_rows += 1

        positions = range(1, num_plots + 1)

        # adjust the figure size to fit all the plots
        if num_rows > 1:
            new_figsize = (figsize[0], figsize[1] * num_rows)
            fig = plt.figure(1, figsize=new_figsize)
        else:
            if num_cols != 1:
                figsize = ((figsize[0] / viz_per_row) * num_cols, figsize[1])
            fig = plt.figure(1, figsize=figsize)

        for i, eig_vec in enumerate(eigenvector_indices):

            ax = fig.add_subplot(num_rows, num_cols, positions[i])

            ax.set_title(
                f"λ_{component} = {round(eigenvals[eig_vec], round_sig_fig)}"
            )

            flow = U[:, eig_vec]

            if round_fig:
                flow = np.round(flow, round_sig_fig)

            self.draw_network(edge_flow=flow, ax=ax, with_labels=with_labels)

        plt.tight_layout()
        plt.show()
