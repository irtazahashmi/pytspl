from pytspl import generate_random_simplicial_complex


class TestSCGenerator:

    def test_generate_random_simplicial_complex(self):
        num_of_nodes = 10
        p = 0.4
        dist_threshold = 0.5
        seed = 42

        sc, coordinates = generate_random_simplicial_complex(
            num_of_nodes=num_of_nodes,
            p=p,
            dist_threshold=dist_threshold,
            seed=seed,
        )

        assert sc.nodes == list(range(num_of_nodes))
        assert len(sc.edges) > 0
        assert len(sc.simplices) > 0
        assert len(coordinates) == num_of_nodes
