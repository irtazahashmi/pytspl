import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

"""Module to read incidence matrices B1 and B2 and process it to an edge list."""


class ReadIncidenceMatrix:
    def __init__(self, B1, B2):
        self.B1 = B1.values
        self.B2 = B2.values
        self.graph = self.creat_nx_graph()

    def number_of_nodes(self):
        return self.B1.shape[0]

    def number_of_edges(self):
        return self.B1.shape[1]

    def number_of_triangles(self):
        return self.B2.shape[0]

    def get_data_summary(self):
        number_of_nodes = self.number_of_nodes()
        number_of_edges = self.number_of_edges()
        number_of_triangles = self.number_of_triangles()
        print("number of nodes: ", number_of_nodes)
        print("number of edges: ", number_of_edges)
        print("number of triangles: ", number_of_triangles)

    def get_adjacency_matrix(self):
        edges = self.number_of_edges()
        nodes = self.number_of_nodes()
        assert edges > 0
        assert nodes > 0

        adjacency = [[0] * nodes for _ in range(nodes)]

        for edge in range(edges):
            a, b = -1, -1
            node = 0
            while node < nodes and a == -1:
                if self.B1[node][edge] != 0:
                    a = node
                node += 1

            while node < nodes and b == -1:
                if self.B1[node][edge] != 0:
                    b = node
                node += 1

            if b == -1:
                b = a

            adjacency[a][b] = -1
            adjacency[b][a] = 1

        return np.array(adjacency)

    def creat_nx_graph(self):
        adjacency = self.get_adjacency_matrix()
        G = nx.from_numpy_matrix(np.array(adjacency))
        return G

    def draw_graph(self):
        nx.draw(self.graph, with_labels=True)
        plt.show()

    def get_nodes(self):
        return self.graph.nodes()

    def get_edge_list(self, rank=None):
        if rank == 1:
            return self.graph.edges()
        elif rank == 2:
            return self.get_triangle_nodes()
        else:
            return list(self.graph.edges()) + list(self.get_triangle_nodes())

    def get_triangle_nodes(self):
        edges = list(self.graph.edges())
        num_of_tri = self.number_of_triangles()
        tri_nodes = []

        for i in range(num_of_tri):
            # find triangles
            nonzeros = np.nonzero(self.B2[i])[0]
            # find nodes
            nodes = [edges[node] for node in nonzeros]
            nodes = [item for tuple_item in nodes for item in tuple_item]
            tri_nodes.append(list(set(nodes)))

        return tri_nodes
