from sclibrary.extended_graph import ExtendedGraph
from sclibrary.network_reader import NetworkReader as nv


class TestNetworkReader:

    def test_read_csv(self):
        filename = "data/sample_data/edges.csv"
        delimeter = " "
        src_col = "Source"
        dest_col = "Target"
        feature_cols = ["Distance"]
        g = nv.read_csv(filename, delimeter, src_col, dest_col, feature_cols)
        assert isinstance(g, ExtendedGraph)