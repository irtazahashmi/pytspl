from sclibrary.io.transportation_loader import (
    get_dataset_summary,
    list_transportation_datasets,
)


class TestTransportationLoader:

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
