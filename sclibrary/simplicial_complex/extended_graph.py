import networkx as nx

from sclibrary.simplicial_complex import SimplicialComplexNetwork


class ExtendedGraph(nx.Graph):
    """Extended Graph module. Built on top of networkx.Graph."""

    def __init__(self, incoming_graph_data=None, **attr):
        """Initialize the ExtendedGraph class using networkx.Graph."""
        super().__init__(incoming_graph_data, **attr)

    def summary(self) -> dict:
        """Provide a summary of the network data."""
        return {
            "number_of_nodes": self.number_of_nodes(),
            "number_of_edges": self.number_of_edges(),
            "is_directed": self.is_directed(),
            "graph_density": nx.density(self),
        }

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
        cliques = nx.enumerate_all_cliques(self)
        triangle_nodes = [x for x in cliques if len(x) == 3]

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
        dist_col_name="distance",
        dist_threshold: float = 1.5,
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
        cliques = nx.enumerate_all_cliques(self)
        # remove 3 nodes cliques
        simplicies = [x for x in cliques if len(x) <= 2]

        # add 2-simplicies based on condition
        if condition == "all":
            simplicies.extend(self.triangles())
        else:
            simplicies.extend(
                self.triangles_dist_based(dist_col_name, dist_threshold)
            )

        return simplicies

    def to_simplicial_complex(
        self,
        condition: str = "all",
        dist_col_name="distance",
        dist_threshold: float = 1.5,
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
        simplices = self.simplicies(
            condition=condition,
            dist_col_name=dist_col_name,
            dist_threshold=dist_threshold,
        )

        node_features = {node: self.nodes[node] for node in self.nodes}
        edge_features = {(u, v): self.edges[u, v] for u, v in self.edges}

        sc = SimplicialComplexNetwork(
            simplices=simplices,
            node_features=node_features,
            edge_features=edge_features,
        )

        return sc
