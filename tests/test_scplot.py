import matplotlib.pyplot as plt
import numpy as np
import pytest

from sclibrary import SCPlot
from sclibrary.simplicial_complex import SimplicialComplex


@pytest.fixture
def sc_plot(sc_mock: SimplicialComplex, coordinates_mock: np.ndarray):
    yield SCPlot(simplical_complex=sc_mock, coordinates=coordinates_mock)


@pytest.fixture
def sc_plot_no_coords(sc_mock: SimplicialComplex):
    yield SCPlot(simplical_complex=sc_mock, coordinates=None)


class TestSCPlot:

    def test_init_axes_coords(self, sc_plot: SCPlot):
        # Create a mock axes object
        _, ax = plt.subplots()

        # Call the _init_axes method
        layout = sc_plot._init_axes(ax)

        # Check if the axis limits are set correctly
        assert ax.get_xlim() == (-1.1, 1.1)
        assert ax.get_ylim() == (-2.625, 0.125)

        # Check if the layout is set correctly
        assert layout == sc_plot.pos

        # Check if the ticks and axis are turned off
        assert ax.get_xaxis().get_ticklocs().size == 0
        assert ax.get_yaxis().get_ticklocs().size == 0
        assert ax.axis("off")

    def test_init_axes_no_coords(self, sc_plot_no_coords: SCPlot):
        # Create a mock axes object
        _, ax = plt.subplots()

        # Call the _init_axes method
        layout = sc_plot_no_coords._init_axes(ax)

        # Check if the axis limits are set correctly
        assert ax.get_xlim() == (-1.1, 1.1)
        assert ax.get_ylim() == (-1.1, 1.1)

        # Check if the layout is nx.spring_layout
        assert layout == sc_plot_no_coords.pos

        # Check if the ticks and axis are turned off
        assert ax.get_xaxis().get_ticklocs().size == 0
        assert ax.get_yaxis().get_ticklocs().size == 0
        assert ax.axis("off")

    def test_create_edge_flow(self, sc_plot: SCPlot, f0_mock: np.ndarray):
        # Call the create_edge_flow method
        edge_flow = sc_plot.create_edge_flow(flow=f0_mock)
        # Check if the edge flow dictionary is created correctly
        assert isinstance(edge_flow, dict)
        assert len(edge_flow) == len(sc_plot.sc.edges)

    def test_draw_sc_nodes(self, sc_plot: SCPlot):
        # Set up the plot
        fig, ax = plt.subplots()

        # Call the draw_sc_nodes method
        sc_plot.draw_sc_nodes(
            node_size=200,
            node_color="red",
            node_edge_colors="black",
            font_size=12,
            font_color="k",
            font_weight="normal",
            cmap=None,
            vmin=None,
            vmax=None,
            alpha=0.8,
            margins=None,
            with_labels=True,
            ax=ax,
        )

        # Add some assertions to check if the plot is generated correctly
        assert ax.get_xlim() == (-1.1, 1.1)
        assert ax.get_ylim() == (-2.625, 0.125)
        assert len(ax.collections) == 1
        assert len(ax.texts) == len(sc_plot.sc.nodes)

        # Close the plot
        plt.close(fig)

    def test_draw_node_labels(self, sc_plot: SCPlot):
        # Create a mock axes object
        ax = plt.gca()

        # Call the method to be tested
        sc_plot._draw_node_labels(
            font_size=12,
            font_color="k",
            font_weight="normal",
            alpha=0.8,
        )

        # Check if the labels are correctly drawn
        for node_id in sc_plot.sc.nodes:
            (x, y) = sc_plot.pos[node_id]
            text = ax.texts[node_id]
            assert text.get_text() == str(node_id)
            assert text.get_position() == (x, y)
            assert text.get_fontsize() == 12
            assert text.get_color() == "k"
            assert text.get_fontweight() == "normal"
            assert text.get_alpha() == 0.8

    def test_draw_edge_labels(self, sc_plot: SCPlot):
        import networkx as nx

        # Create a graph
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 0)])

        # Set up the plot
        fig, ax = plt.subplots()
        # Create edge labels
        edge_labels = {(0, 1): "A", (1, 2): "B", (2, 0): "C"}
        # Draw the edge labels
        sc_plot.draw_edge_labels(edge_labels, ax=ax)

        # Assert that the edge labels are displayed correctly
        for (src, dest), label in edge_labels.items():
            assert ax.texts[src].get_text() == label

        plt.close(fig)

    def test_draw_network_without_flow(self, sc_plot: SCPlot):
        # Test drawing the network without flow
        fig, ax = plt.subplots()
        sc_plot.draw_network(edge_flow=None, ax=ax)
        # Assert that the plot is not empty
        assert ax.lines is not None

        plt.close(fig)

    def test_draw_network_with_flow(
        self, sc_plot: SCPlot, f0_mock: np.ndarray
    ):
        # Test drawing the network with flow
        edge_flow = sc_plot.create_edge_flow(flow=f0_mock)
        fig, ax = plt.subplots()
        sc_plot.draw_network(edge_flow=edge_flow, ax=ax)
        # Assert that the plot is not empty
        assert ax.lines is not None

        plt.close(fig)
