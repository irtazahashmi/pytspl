import networkx as nx

"""Extended Graph module. Built on top of networkx.Graph."""


class ExtendedGraph(nx.Graph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    def triangles(self) -> list:
        """
        Returns a list of triangles in the graph.

        Returns:
            list: List of triangles.
        """
        cliques = nx.enumerate_all_cliques(self)
        triangle_nodes = [x for x in cliques if len(x) == 3]
        return triangle_nodes

    def triangles_dist_based(self, dist_col_name: str, epsilon: float) -> list:
        """
        Returns a list of triangles in the graph that satisfy the condition:
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
        Returns a list of simplicies in the graph.

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
