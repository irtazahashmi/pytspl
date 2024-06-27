from pytspl.hogde_gp.kernels import Kernels


class TestKernels:

    def test_get_names(self):
        names = Kernels.get_names()
        expected_names = [
            "diffusion",
            "diffusion_non_hc",
            "matern",
            "matern_non_hc",
            "laplacian",
            "laplacian_non_hc",
        ]
        assert names == expected_names

    def test_get_forex_kernels(self):
        forex_kernels = Kernels.get_forex_kernerls()

        assert isinstance(forex_kernels, dict)
        assert len(forex_kernels) == 5
