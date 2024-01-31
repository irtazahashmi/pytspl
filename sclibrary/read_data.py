import networkx as nx
import numpy as np
import pandas as pd

from sclibrary.extended_graph import ExtendedGraph

"""Module for reading graph data."""


class GraphDataReader:
    @staticmethod
    def read_csv(
        filename: str,
        delimeter: str,
        src_col: str,
        dest_col: str,
        weight_col: str = "",
    ) -> ExtendedGraph:
        """
        Reads a csv file and returns a graph.

        Args:
            filename (str): The name of the csv file.
            delimeter (str): The delimeter used in the csv file.
            src_col (str): The name of the column containing the source nodes.
            dest_col (str): The name of the column containing the destination nodes.
            weight_col (str, optional): The name of the weight column. Defaults to "".

        Returns:
            ExtendedGraph: The graph read from the csv file.
        """
        df = pd.read_csv(filename, sep=delimeter)

        # Create a graph
        G = nx.Graph()
        # add edges
        for _, row in df.iterrows():
            G.add_edge(row[src_col], row[dest_col])
        # add weights if any
        if weight_col:
            for _, row in df.iterrows():
                G[row[src_col]][row[dest_col]]["weight"] = row[weight_col]

        return ExtendedGraph(G)

    @staticmethod
    def read_incidence_matrix(
        B1_filename: str, B2_filename: str
    ) -> ExtendedGraph:
        """
        Reads the B1 and B2 incidence matrix files.

        Args:
            B1_filename (str): The name of the B1 incidence matrix file.
            B2_filename (str): The name of the B2 incidence matrix file.

        Returns:
            ExtendedGraph: The graph read from the incidence matrix files.
        """
        B1 = pd.read_csv(B1_filename, header=None).values
        B2 = pd.read_csv(B2_filename, header=None).values

        # create adjacency matrix
        nodes = B1.shape[0]
        edges = B1.shape[1]
        assert edges > 0
        assert nodes > 0

        adjacency = [[0] * nodes for _ in range(nodes)]

        for edge in range(edges):
            a, b = -1, -1
            node = 0

            while node < nodes and a == -1:
                if B1[node][edge] != 0:
                    a = node
                node += 1

            while node < nodes and b == -1:
                if B1[node][edge] != 0:
                    b = node
                node += 1

            if b == -1:
                b = a

            adjacency[a][b] = -1
            adjacency[b][a] = 1

        # create graph
        G = nx.from_numpy_matrix(np.array(adjacency))
        return ExtendedGraph(G)

    @staticmethod
    def get_coordinates(
        filename: str,
        node_id_col: str,
        x_col: str,
        y_col: str,
        delimeter: str = ",",
    ) -> dict:
        """
        Reads a csv file and returns a dictionary of coordinates.

        Args:
            filename (str): The name of the csv file.
            node_id_col (str): The name of the column containing the node ids.
            x_col (str): The name of the column containing the x coordinates.
            y_col (str): The name of the column containing the y coordinates.
            delimeter (str, optional): The delimeter used in the csv file. Defaults to ",".

        Returns:
            dict: A dictionary of coordinates (node_id : (x, y)).
        """
        coordinates = pd.read_csv(filename, sep=delimeter)
        # create a dictionary of coordinates (node_id : (x, y))
        return dict(
            zip(
                coordinates[node_id_col],
                zip(coordinates[x_col], coordinates[y_col]),
            )
        )
