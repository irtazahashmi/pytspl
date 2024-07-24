import pickle

import numpy as np
import pkg_resources

from pytspl.simplicial_complex.scbuilder import SCBuilder

WSN_DATA_FOLDER = pkg_resources.resource_filename("pytspl", "data/wsn")


def load_wsn_data() -> tuple:
    """
    Load the water supply network data and return the simplicial complex
    and coordinates.

    Returns:
        tuple:
            SimplicialComplex: The simplicial complex of the water supply
            network data.
            dict: The coordinates of the nodes. If the coordinates do not
            exist, the coordinates are generated using spring layout.
            np.ndarray: The flow data of the water supply
    """

    with open(f"{WSN_DATA_FOLDER}/water_network.pkl", "rb") as f:
        B1, flow_rate, _, head, hr = pickle.load(f)

    num_edges = B1.shape[1]
    nodes = set()
    edges = []

    for j in range(num_edges):
        col = B1[:, j]
        from_node = np.where(col == -1)[0][0]
        to_node = np.where(col == 1)[0][0]

        nodes.add(from_node)
        nodes.add(to_node)

        edges.append((from_node, to_node))

    nodes = list(range(max(nodes) + 1))

    sc = SCBuilder(nodes=nodes, edges=edges).to_simplicial_complex()

    # no coordinates
    coordinates = sc.generate_coordinates()

    # read flow data
    hr = hr.squeeze()
    head = np.asarray(head)

    flow_rate = np.asarray(flow_rate)
    sign = np.sign(flow_rate)
    flow_rate = -hr * sign * np.abs(flow_rate) ** 1.852
    hr[:] = 1

    y = np.concatenate((head, flow_rate))

    return sc, coordinates, y
