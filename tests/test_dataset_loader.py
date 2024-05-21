import numpy as np

from sclibrary import SimplicialComplex
from sclibrary.io.dataset_loader import (
    get_dataset_summary,
    list_transportation_datasets,
    load,
    load_paper_data,
)
from sclibrary.io.network_reader import read_B1, read_B2


class TestDatasetLoader:

    def test_list_list_transportation_datasets(self):
        datasets = list_transportation_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0

    def test_get_dataset_summary(self):
        dataset = "siouxfalls"
        summary = get_dataset_summary(dataset)
        assert isinstance(summary, dict)
        assert "number_of_zones" in summary
        assert summary["number_of_zones"] == "24"
        assert "number_of_nodes" in summary
        assert summary["number_of_nodes"] == "24"
        assert "first_thru_node" in summary
        assert summary["first_thru_node"] == "1"
        assert "number_of_links" in summary
        assert summary["number_of_links"] == "76"
        assert "features" in summary
        assert len(summary["features"]) == 10
        assert "coordinates_exist" in summary
        assert summary["coordinates_exist"] is True
        assert "flow_data_exist" in summary
        assert summary["flow_data_exist"] is True

    def test_get_dataset_summary_no_coordinates_flow(self):
        # test dataset does not have coordinates and flow data
        dataset = "test_dataset"
        summary = get_dataset_summary(dataset)
        assert isinstance(summary, dict)
        assert summary["coordinates_exist"] is False
        assert "flow_data_exist" in summary
        assert summary["flow_data_exist"] is False

    def test_load_dataset(self):
        dataset = "siouxfalls"

        sc, coordinates, flow_dict = load(dataset=dataset)

        assert isinstance(sc, SimplicialComplex)
        assert isinstance(coordinates, dict)
        assert isinstance(flow_dict, dict)

        # nodes should be equal to sc nodes
        assert sc.nodes == list(coordinates.keys())
        # sc edges should be equal to flow edges
        assert sc.edges == list(flow_dict.keys())

    def test_load_dataset_no_coordinates(self):
        # coordinates are generated from nx.spring_layout
        dataset = "test_dataset"
        _, coordinates, _ = load(dataset)

        assert isinstance(coordinates, dict)
        assert len(coordinates) > 0

    def test_load_transportation_datasets(self):

        datasets_with_all_data = [
            "anaheim",
            "chicago-regional",
            "chicago-sketch",
            "siouxfalls",
        ]

        for dataset in datasets_with_all_data:

            sc, coordinates, flow_dict = load(dataset=dataset)

            assert isinstance(sc, SimplicialComplex)
            assert isinstance(coordinates, dict)
            assert isinstance(flow_dict, dict)

            # coordinate nodes should be equal to sc nodes
            assert sc.nodes == list(coordinates.keys())
            # number of edges should be equal to the number of
            # flow data
            assert sc.edges == list(flow_dict.keys())

    def test_load_paper_data(self):
        import pandas as pd

        B1 = pd.read_csv("data/paper_data/B1.csv", header=None).to_numpy()
        B2 = pd.read_csv("data/paper_data/B2t.csv", header=None).to_numpy().T

        sc, coordinates = load_paper_data()

        assert isinstance(sc, SimplicialComplex)
        assert isinstance(coordinates, dict)

        B1_calculated = sc.incidence_matrix(rank=1)
        B2_calculated = sc.incidence_matrix(rank=2)

        print(B1)
        print("Me", B1_calculated)

        assert np.array_equal(B1, B1_calculated)
        assert np.array_equal(B2, B2_calculated)

        # coordinate nodes should be equal to sc nodes
        assert sc.nodes == list(coordinates.keys())
