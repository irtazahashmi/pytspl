"""SC builder module to build simplicial complex networks using
0-simplicies (nodes), 1-simplices (edges) and 2-simplicies (triangles).

The 2-simplices can be added in three ways:
    - Triangles passed as an argument.
    - All triangles in the simplicial complex.
    - Triangles based on a condition e.g. distance.
"""

import networkx as nx

from pytspl.simplicial_complex import SimplicialComplex


class SCBuilder:
    """SCBuilder is used to build a simplicial complex by defining the
    2-simplices using different ways."""

    def __init__(
        self,
        nodes: list,
        edges: list,
        node_features: dict = {},
        edge_features: dict = {},
    ):
        """Initialize the SCBuilder object."""
        # 0-simplicies - nodes
        self.nodes = nodes
        # 1-simplicies - edges
        self.edges = edges

        # node and edge features
        self.node_features = node_features
        self.edge_features = edge_features

    def triangles(self) -> list:
        """
        Get a list of triangles in the graph.

        Returns:
            list: List of triangles.
        """
        g = nx.Graph()
        g.add_edges_from(self.edges)
        cliques = nx.enumerate_all_cliques(g)
        triangle_nodes = [x for x in cliques if len(x) == 3]
        # sort the triangles
        triangle_nodes = [sorted(tri) for tri in triangle_nodes]
        return triangle_nodes

    def triangles_dist_based(self, dist_col_name: str, epsilon: float) -> list:
        """
        Get a list of triangles in the graph that satisfy the condition:
            d(a, b) < epsilon, d(a, c) < epsilon, d(b, c) < epsilon

        Args:
            dist_col_name (str): Name of the column that contains the distance.
            epsilon (float, optional): Distance threshold to consider for
            triangles.

        Returns:
            list: List of triangles that satisfy the condition.
        """
        triangle_nodes = self.triangles()

        conditional_tri = []
        for a, b, c in triangle_nodes:
            if (
                self.edge_features[(a, b)][dist_col_name]
                and self.edge_features[(b, c)][dist_col_name]
                and self.edge_features[(a, c)][dist_col_name]
            ):
                dist_ab = self.edge_features[(a, b)][dist_col_name]
                dist_ac = self.edge_features[(b, c)][dist_col_name]
                dist_bc = self.edge_features[(a, c)][dist_col_name]

                if (
                    dist_ab < epsilon
                    and dist_ac < epsilon
                    and dist_bc < epsilon
                ):
                    conditional_tri.append([a, b, c])

        return conditional_tri

    def to_simplicial_complex(
        self,
        condition: str = "all",
        dist_col_name: str = "distance",
        dist_threshold: float = 1.5,
        triangles=None,
    ) -> SimplicialComplex:
        """
        Convert the graph to a simplicial complex using the given condition
        of simplicies. The simplicial complex will also have node and edge
        features.

        Args:
            condition (str, optional): Condition to build the 2-simplicies
            (triangles). Defaults to "all".
            Options:
            - "all": All simplicies.
            - "distance": Based on distance.

            dist_col_name (str, optional): Name of the column that contains
            the distance.
            dist_threshold (float, optional): Distance threshold to consider
            for simplicies. Defaults to 1.5.

        Returns:
            SimplicialComplex: Simplicial complex network.
        """
        if triangles is None:
            if condition == "all":
                # add all 2-simplicies
                triangles = self.triangles()
            else:
                # add 2-simplicies based on condition
                triangles = self.triangles_dist_based(
                    dist_col_name=dist_col_name, epsilon=dist_threshold
                )

        # create the simplicial complex
        sc = SimplicialComplex(
            nodes=self.nodes,
            edges=self.edges,
            triangles=triangles,
            node_features=self.node_features,
            edge_features=self.edge_features,
        )

        return sc
