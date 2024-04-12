from sclibrary import read_csv
from sclibrary.simplicial_complex.extended_graph import ExtendedGraph


class TestNetworkReader:

    def test_read_csv(self):
        filename = "data/sample_data/edges.csv"
        delimeter = " "
        src_col = "Source"
        dest_col = "Target"
        feature_cols = ["Distance"]
        g = read_csv(filename, delimeter, src_col, dest_col, feature_cols)
        assert isinstance(g, ExtendedGraph)
