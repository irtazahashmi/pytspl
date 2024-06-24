import gpytorch
import torch

from sclibrary.hogde_gp.exact_gp import ExactGPModel


class TestExactGPModel:

    def test_forward_pass_zero_mean(self):
        train_x = torch.tensor(4.5)
        train_y = torch.tensor(1)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        kernel = gpytorch.kernels.Kernel()

        exact_gp = ExactGPModel(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            kernel=kernel,
            mean_function="zero",
        )

        x = torch.tensor([])
        res = exact_gp.forward(x=x)
        assert isinstance(res, gpytorch.distributions.MultivariateNormal)

    def test_forward_pass_constant_mean(self):
        train_x = torch.tensor(4.5)
        train_y = torch.tensor(1)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        kernel = gpytorch.kernels.Kernel()

        exact_gp = ExactGPModel(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            kernel=kernel,
        )

        x = torch.tensor([])
        res = exact_gp.forward(x=x)
        assert isinstance(res, gpytorch.distributions.MultivariateNormal)
