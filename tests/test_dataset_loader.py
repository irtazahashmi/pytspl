import numpy as np
import pandas as pd
import pytest

from sclibrary import SimplicialComplex
from sclibrary.io.dataset_loader import list_datasets, load


class TestDatasetLoader:

    def test_list_datasets(self):
        datasets = list_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0

    def test_load_dataset_paper(self):
        B1 = pd.read_csv("data/paper_data/B1.csv", header=None).to_numpy()
        B2 = pd.read_csv("data/paper_data/B2t.csv", header=None).to_numpy().T

        dataset = "paper"

        sc, coordinates = load(dataset=dataset)

        assert isinstance(sc, SimplicialComplex)
        assert isinstance(coordinates, dict)

        B1_calculated = sc.incidence_matrix(rank=1).toarray()
        B2_calculated = sc.incidence_matrix(rank=2).toarray()

        assert np.array_equal(B1, B1_calculated)
        assert np.array_equal(B2, B2_calculated)

        # coordinate nodes should be equal to sc nodes
        assert sc.nodes == list(coordinates.keys())

    def test_load_unknown_dataset(self):
        dataset = "unknown"
        with pytest.raises(ValueError):
            load(dataset=dataset)

    def test_load_dataset_forex(self):
        dataset = "forex"

        sc, coordinates, flow = load(dataset=dataset)

        assert isinstance(sc, SimplicialComplex)
        assert isinstance(coordinates, dict)
        assert isinstance(flow, dict)

    def test_load_dataset_transportation(self):
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
