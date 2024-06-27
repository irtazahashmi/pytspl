from sclibrary.io.data_loaders.transportation_loader import (
    list_transportation_datasets,
)


class TestTransportationLoader:

    def test_list_list_transportation_datasets(self):
        datasets = list_transportation_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0
