import numpy as np
import pytest

from sclibrary import read_csv


@pytest.fixture
def graph():
    filename = "data/paper_data/edges.csv"
    delimeter = " "
    src_col = "Source"
    dest_col = "Target"
    feature_cols = ["Distance"]
    g = read_csv(filename, delimeter, src_col, dest_col, feature_cols)
    yield g


class TestExtendedGraph:
    def test_triangles(self, graph):
        triangles = graph.triangles()
        expected_triangles = np.array([[0, 1, 2], [0, 2, 3], [5, 4, 6]])
        assert np.array_equal(triangles, expected_triangles)

    def test_triangles_dist_based(self, graph):
        eps = 1.5
        triangles = graph.triangles_dist_based(
            dist_col_name="Distance", epsilon=eps
        )
        expected_triangles = np.array([[0, 1, 2], [5, 4, 6]])
        assert np.array_equal(triangles, expected_triangles)

    def test_all_simplicies(self, graph):
        condition = "all"
        simplicies = graph.simplicies(condition=condition)
        print(simplicies)
        expected_simplicies = [
            [0],
            [1],
            [2],
            [3],
            [5],
            [4],
            [6],
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [2, 3],
            [2, 5],
            [3, 4],
            [5, 4],
            [5, 6],
            [4, 6],
            [0, 1, 2],
            [0, 2, 3],
            [5, 4, 6],
        ]

        assert simplicies == expected_simplicies

    def test_conditional_dist_simplicies(self, graph):
        condition = "distance"
        dist_col_name = "Distance"
        dist_threshold = 1.5
        simplicies = graph.simplicies(
            condition=condition,
            dist_col_name=dist_col_name,
            dist_threshold=dist_threshold,
        )
        expected_simplicies = [
            [0],
            [1],
            [2],
            [3],
            [5],
            [4],
            [6],
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [2, 3],
            [2, 5],
            [3, 4],
            [5, 4],
            [5, 6],
            [4, 6],
            [0, 1, 2],
            [5, 4, 6],
        ]

        assert simplicies == expected_simplicies
