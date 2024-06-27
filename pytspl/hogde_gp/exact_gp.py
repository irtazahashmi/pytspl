"""Module for the ExactGPModel class."""

import gpytorch
import torch
from gpytorch.constraints import Positive


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.tensor,
        train_y: torch.tensor,
        likelihood: gpytorch.likelihoods,
        kernel: gpytorch.kernels.Kernel,
        mean_function=None,
    ):
        """
        Initialize the ExactGPModel class.

        Args:
            train_x (torch.tensor): The training data.
            train_y (torch.tensor): The training labels.
            likelihood (gpytorch.likelihoods): The likelihood function.
            kernel (gpytorch.kernels.Kernel): The kernel function.
            mean_function (_type_, optional): The mean function.
            Defaults to None.
        """
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        if mean_function == "zero":
            self.mean_module = gpytorch.means.ZeroMean()
        else:
            self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel, outputscale_constraint=Positive()
        )

    def forward(self, x: torch.tensor):
        """
        Forward pass for the model.

        Args:
            x (torch.tensor): The input data.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
