import networkx as nx

"""Extended Graph module. Built on top of networkx.Graph."""


class ExtendedGraph(nx.Graph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    @property
    def triangles(self) -> list:
        """
        Returns a list of triangles in the graph.

        Returns:
            list: List of triangles.
        """
        cliques = nx.enumerate_all_cliques(self)
        triangle_nodes = [x for x in cliques if len(x) == 3]
        return triangle_nodes

    @property
    def triangles_based_on_distance(self, epsilon: float = 1.5) -> list:
        """
        Returns a list of triangles in the graph that satisfy the condition:
            d(a, b) < epsilon, d(a, c) < epsilon, d(b, c) < epsilon

        Args:
            epsilon (float, optional): Distance threshold. Defaults to 1.5.

        Returns:
            list: List of triangles that satisfy the condition.
        """
        cliques = nx.enumerate_all_cliques(self)
        triangle_nodes = [x for x in cliques if len(x) == 3]

        conditional_tri = []
        for a, b, c in triangle_nodes:
            if (
                self.get_edge_data(a, b)
                and self.get_edge_data(b, c)
                and self.get_edge_data(a, c)
            ):
                dist_ab = self.get_edge_data(a, b)["weight"]
                dist_ac = self.get_edge_data(a, c)["weight"]
                dist_bc = self.get_edge_data(b, c)["weight"]

                if (
                    dist_ab < epsilon
                    and dist_ac < epsilon
                    and dist_bc < epsilon
                ):
                    conditional_tri.append([a, b, c])

        return conditional_tri

    def simplicies(self, conditional_triangles=True) -> list:
        """
        Returns a list of simplicies in the graph.

        Args:
            conditional_triangles (bool, optional): If True, returns triangles that satisfy the condition. Defaults to True.

        Returns:
            list: List of simplicies.
        """
        cliques = nx.enumerate_all_cliques(self)
        # remove 3 nodes cliques
        simplicies = [x for x in cliques if len(x) <= 2]
        # add triangles based on condition

        if conditional_triangles:
            simplicies = simplicies + self.triangles_based_on_distance
        else:
            simplicies = simplicies + self.triangles

        return simplicies
