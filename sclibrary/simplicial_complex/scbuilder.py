import networkx as nx

from sclibrary.simplicial_complex import SimplicialComplexNetwork


class SCBuilder(nx.Graph):
    """SC builder module. Built on top of networkx.Graph."""

    def __init__(self, incoming_graph_data=None, **attr):
        """Initialize the ExtendedGraph class using networkx.Graph."""
        super().__init__(incoming_graph_data, **attr)

    def triangles(self) -> list:
        """
        Get a list of triangles in the graph.

        Returns:
            list: List of triangles.
        """
        cliques = nx.enumerate_all_cliques(self)
        triangle_nodes = [x for x in cliques if len(x) == 3]
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
                self.get_edge_data(a, b)[dist_col_name]
                and self.get_edge_data(b, c)[dist_col_name]
                and self.get_edge_data(a, c)[dist_col_name]
            ):
                dist_ab = self.get_edge_data(a, b)[dist_col_name]
                dist_ac = self.get_edge_data(a, c)[dist_col_name]
                dist_bc = self.get_edge_data(b, c)[dist_col_name]

                if (
                    dist_ab < epsilon
                    and dist_ac < epsilon
                    and dist_bc < epsilon
                ):
                    conditional_tri.append([a, b, c])

        return conditional_tri

    def simplicies(
        self,
        condition: str = "all",
        dist_col_name: str = "distance",
        dist_threshold: float = 1.5,
        triangles=None,
    ) -> list:
        """
        Get a list of simplicies in the graph.

        Args:
            condition (str, optional): Condition to filter simplicies.
            Defaults to "all".
            Options:
                - "all": All simplicies.
                - "distance": Based on distance.

            dist_col_name (str, optional): Name of the column that contains
            the distance.
            dist_threshold (float, optional): Distance threshold to consider
            for simplicies. Defaults to 1.5.

        Returns:
            list: List of simplicies.
        """
        # 0-simplicies - nodes
        nodes = [[node] for node in self.nodes()]
        # 1-simplicies - edges
        edges = [list(edge) for edge in self.edges()]
        simplices = nodes + edges

        # add 2-simplicies based on given triangles
        if triangles is not None:
            simplices.extend(triangles)
        elif condition == "all":
            # add all 2-simplicies
            simplices.extend(self.triangles())
        else:
            # add 2-simplicies based on condition
            simplices.extend(
                self.triangles_dist_based(dist_col_name, dist_threshold)
            )

        return simplices

    def to_simplicial_complex(
        self,
        condition: str = "all",
        dist_col_name: str = "distance",
        dist_threshold: float = 1.5,
        triangles=None,
    ):
        """
        Convert the graph to a simplicial complex using the given condition
        of simplicies. The simplicial complex will also have node and edge
        features.

        Args:
            condition (str, optional): Condition to filter simplicies.
            Defaults to "all".
            Options:
                - "all": All simplicies.
                - "distance": Based on distance.

            dist_col_name (str, optional): Name of the column that contains
            the distance.
            dist_threshold (float, optional): Distance threshold to consider
            for simplicies. Defaults to 1.5.

        Returns:
            SimplicialComplexNetwork: Simplicial complex network.
        """
        # get simplicies
        simplices = self.simplicies(
            condition=condition,
            dist_col_name=dist_col_name,
            dist_threshold=dist_threshold,
            triangles=triangles,
        )

        node_features = {node: self.nodes[node] for node in self.nodes}
        edge_features = {(u, v): self.edges[u, v] for u, v in self.edges}

        sc = SimplicialComplexNetwork(
            simplices=simplices,
            node_features=node_features,
            edge_features=edge_features,
        )

        return sc
