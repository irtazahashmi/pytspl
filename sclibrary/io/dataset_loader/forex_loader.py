import networkx as nx
import pandas as pd

from sclibrary.io.network_reader import read_B1_B2


def load_forex_data() -> tuple:
    """
    Load the forex data and return the simplicial complex and coordinates.

    Returns:
        tuple:
            SimplicialComplex: The simplicial complex of the forex data.
            dict: The coordinates of the nodes. If the coordinates do not
            exist, the coordinates are generated using spring layout.
            dict: The flow data of the forex data.
    """
    data_folder = "data/foreign_exchange"
    B1_filename = f"{data_folder}/B1.csv"
    B2_filename = f"{data_folder}/B2t.csv"
    y_filename = f"{data_folder}/flow_FX_1538755200.csv"

    scbuilder, triangles = read_B1_B2(
        B1_filename=B1_filename, B2_filename=B2_filename
    )
    sc = scbuilder.to_simplicial_complex(triangles=triangles)

    # no coordinates for forex data - generate using spring layout
    print("Generating coordinates using spring layout.")
    graph = nx.Graph(sc.edges)
    coordinates = nx.spring_layout(graph)

    flow = pd.read_csv(y_filename, header=None).values[:, 0]
    flow = {edge: flow[i] for i, edge in enumerate(sc.edges)}

    return sc, coordinates, flow
