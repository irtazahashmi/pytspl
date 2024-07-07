import numpy as np
import pytest

from pytspl import read_csv
from pytspl.simplicial_complex.scbuilder import SCBuilder


@pytest.fixture
def graph():
    data_folder = "data/paper_data"
    filename = data_folder + "/edges.csv"
    delimeter = " "
    src_col = "Source"
    dest_col = "Target"
    feature_cols = ["Distance"]

    g = read_csv(
        filename=filename,
        delimeter=delimeter,
        src_col=src_col,
        dest_col=dest_col,
        feature_cols=feature_cols,
    )
    yield g


class TestSCBuilder:
    def test_triangles(self, graph: SCBuilder):
        triangles = graph.triangles()
        expected_triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])
        assert np.array_equal(triangles, expected_triangles)

    def test_triangles_dist_based(self, graph: SCBuilder):
        eps = 1.5
        triangles = graph.triangles_dist_based(
            dist_col_name="Distance", epsilon=eps
        )
        expected_triangles = np.array([[0, 1, 2], [4, 5, 6]])
        assert np.array_equal(triangles, expected_triangles)

    def test_condition_all_simplicies(self, graph: SCBuilder):
        condition = "all"
        simplices = graph.to_simplicial_complex(condition=condition).simplices

        expected_simplices = [
            (0,),
            (1,),
            (2,),
            (3,),
            (4,),
            (5,),
            (6,),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (2, 3),
            (2, 5),
            (3, 4),
            (4, 5),
            (4, 6),
            (5, 6),
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
        ]

        assert simplices == expected_simplices

    def test_conditional_dist_simplicies(self, graph: SCBuilder):
        condition = "distance"
        dist_col_name = "Distance"
        dist_threshold = 1.5
        simplices = graph.to_simplicial_complex(
            condition=condition,
            dist_col_name=dist_col_name,
            dist_threshold=dist_threshold,
        ).simplices

        expected_simplicies = [
            (0,),
            (1,),
            (2,),
            (3,),
            (4,),
            (5,),
            (6,),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (2, 3),
            (2, 5),
            (3, 4),
            (4, 5),
            (4, 6),
            (5, 6),
            [0, 1, 2],
            [4, 5, 6],
        ]

        assert simplices == expected_simplicies
