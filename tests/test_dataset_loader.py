import pandas as pd

from sclibrary import SimplicialComplexNetwork
from sclibrary.io.dataset_loader import get_dataset_summary, load, load_flow


class TestDatasetLoader:

    def test_list_list_transportation_datasets(self):
        from sclibrary.io.dataset_loader import list_transportation_datasets

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

    def test_get_dataset_summary_test_dataset(self):
        dataset = "test_dataset"
        summary = get_dataset_summary(dataset)
        assert isinstance(summary, dict)
        assert summary["coordinates_exist"] is False
        assert "flow_data_exist" in summary
        assert summary["flow_data_exist"] is False

    def test_load_flow(self):
        dataset = "siouxfalls"
        flow = load_flow(dataset)
        assert flow is not None
        assert isinstance(flow, pd.DataFrame)
        assert len(flow) > 0

    def test_load_flow_not_found(self):
        dataset = "unknown_dataset"
        flow = load_flow(dataset)
        assert flow.empty is True

    def test_load_dataset(self):
        dataset = "siouxfalls"
        sc, coordinates, flow_dict = load(dataset)
        assert sc is not None
        assert coordinates is not None
        assert flow_dict is not None

        assert isinstance(sc, SimplicialComplexNetwork)
        assert isinstance(coordinates, dict)
        assert isinstance(flow_dict, dict)

        assert len(coordinates) == 24
        assert len(flow_dict) == 38

    def test_load_dataset_no_coordinates(self):
        dataset = "test_dataset"
        _, coordinates, _ = load(dataset)

        assert isinstance(coordinates, dict)
        assert len(coordinates) > 0

    def test_transportation_data_loading(self):
        from sclibrary.io.dataset_loader import list_transportation_datasets

        datasets = list_transportation_datasets()
        skip_datasets = ["test_dataset", "goldcoast"]
        for dataset in datasets:
            if dataset in skip_datasets:
                continue

            sc, coordinates, flow_dict = load(dataset=dataset)
            assert sc is not None
            assert coordinates is not None
            assert flow_dict is not None

            assert isinstance(sc, SimplicialComplexNetwork)
            assert isinstance(coordinates, dict)
            assert isinstance(flow_dict, dict)

            # number of edge flow is equal to number of edges in sc
            flow_len = len(flow_dict)
            mat_len = sc.hodge_laplacian_matrix().shape[0]
            assert flow_len == mat_len
