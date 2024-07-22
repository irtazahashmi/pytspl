"""Module for training the Hodge Gaussian Process model."""

import gpytorch
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torcheval.metrics.functional import r2_score

from pytspl.simplicial_complex import SimplicialComplex

DATA_TYPE = torch.float32


class HodgeGPTrainer:
    def __init__(
        self,
        sc: SimplicialComplex,
        y: np.ndarray,
        output_device: str = "cpu",
    ):
        """
        Initialize the HodgeGPTrainer class.

        Args:
            sc (SimplicialComplex): The simplicial complex.
            y (np.ndarray): The target values.
            output_device (str, optional): The output device for
            the tensors. Defaults to "cpu".
        """
        self.sc = sc
        self.y = y

        self.output_device = torch.device(output_device)

    def get_laplacians(self) -> list:
        """
        Return the Laplacian matrices as a list of tensors.

        Returns:
            list(torch.tensor): The Laplacian matrices.
        """
        L1 = self.sc.hodge_laplacian_matrix(rank=1).toarray()
        L1l = self.sc.lower_laplacian_matrix(rank=1).toarray()
        L1u = self.sc.upper_laplacian_matrix(rank=1).toarray()

        print("L1: ", L1.shape)
        print("L1l: ", L1l.shape)
        print("L1u: ", L1u.shape)

        L1 = torch.tensor(L1, dtype=DATA_TYPE)
        L1_down = torch.tensor(L1l, dtype=DATA_TYPE)
        L1_up = torch.tensor(L1u, dtype=DATA_TYPE)

        laplacians = [L1, L1_down, L1_up]
        laplacians = [
            laplacian.to(self.output_device) for laplacian in laplacians
        ]

        return laplacians

    def get_incidence_matrices(self) -> list:
        """
        Return the incidence matrices as a list of tensors.

        Returns:
            list(torch.tensor): The incidence matrices.
        """
        B1 = self.sc.incidence_matrix(rank=1).toarray()
        B2 = self.sc.incidence_matrix(rank=2).toarray()

        print("B1: ", B1.shape)
        print("B2: ", B2.shape)

        B1 = torch.tensor(B1, dtype=DATA_TYPE)
        B2 = torch.tensor(B2, dtype=DATA_TYPE)

        incidence_matrices = [
            B1.to(self.output_device),
            B2.to(self.output_device),
        ]

        return incidence_matrices

    def get_eigenpairs(self, tolerance: float = 1e-3) -> list:
        """
        Return the eigenpairs of the Laplacian matrices.

        Args:
            tolerance (float, optional): The tolerance for eigenvalues
            to be considered zero. Defaults to 1e-3.

        Returns:
            list(torch.tensor): The eigenpairs of the Laplacian matrices.
        """
        h_eigenvecs, h_eigenvals = self.sc.get_component_eigenpair(
            component="harmonic", tolerance=tolerance
        )
        g_eigenvecs, g_eigenvals = self.sc.get_component_eigenpair(
            component="gradient", tolerance=tolerance
        )
        c_eigenvecs, c_eigenvals = self.sc.get_component_eigenpair(
            component="curl", tolerance=tolerance
        )

        h_eigenvecs = torch.tensor(h_eigenvecs, dtype=DATA_TYPE)
        g_eigenvecs = torch.tensor(g_eigenvecs, dtype=DATA_TYPE)
        c_eigenvecs = torch.tensor(c_eigenvecs, dtype=DATA_TYPE)
        h_eigenvals = torch.tensor(h_eigenvals, dtype=DATA_TYPE)
        g_eigenvals = torch.tensor(g_eigenvals, dtype=DATA_TYPE)
        c_eigenvals = torch.tensor(c_eigenvals, dtype=DATA_TYPE)

        eigenpairs = [
            h_eigenvecs,
            g_eigenvecs,
            c_eigenvecs,
            h_eigenvals,
            g_eigenvals,
            c_eigenvals,
        ]

        eigenpairs = [
            eigenpair.to(self.output_device) for eigenpair in eigenpairs
        ]

        return eigenpairs

    def normalize_data(
        self, y_train: torch.tensor, y_test: torch.tensor, y: torch.tensor
    ) -> tuple:
        """
        Normalize the target values.

        Args:
            y_train (torch.tensor): The training target values.
            y_test (torch.tensor): The testing target values.
            y (torch.tensor): The target values.

        Returns:
            tuple(torch.tensor, torch.tensor, torch.tensor): The normalized
            target values.
        """
        orig_mean, orig_std = torch.mean(y_train), torch.std(y_train)

        y_train = (y_train - orig_mean) / orig_std
        y_test = (y_test - orig_mean) / orig_std
        y = (y - orig_mean) / orig_std

        return y_train, y_test, y

    def train_test_split(
        self,
        train_ratio: float = 0.8,
        data_normalization: bool = False,
        seed: int = 4,
    ) -> tuple:
        """
        Split the data into training and validation sets.

        Args:
            train_ratio (float, optional): The ratio of the training
            data. Defaults to 0.8.
            data_normalization (bool, optional): Whether to normalize
            the target data. Defaults to False.
            seed (int, optional): The random seed. Defaults to 4.

        Returns:
            tuple(torch.tensor, torch.tensor, torch.tensor, torch.tensor,
            torch.tensor, torch.tensor): The training and testing data.
        """

        B1 = self.sc.incidence_matrix(rank=1).toarray()
        y = self.y

        # split the data into training and testing sets
        n1 = B1.shape[1]
        x = np.arange(n1)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=train_ratio, random_state=seed
        )

        # print shapes
        print(f"x_train: {x_train.shape}")
        print(f"x_test: {x_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test: {y_test.shape}")

        # convert to tensors
        x_train, y_train = torch.tensor(
            x_train, dtype=DATA_TYPE
        ), torch.tensor(y_train, dtype=DATA_TYPE)
        x_test, y_test = torch.tensor(x_test, dtype=DATA_TYPE), torch.tensor(
            y_test, dtype=DATA_TYPE
        )
        x, y = torch.tensor(x, dtype=DATA_TYPE), torch.tensor(
            y, dtype=DATA_TYPE
        )

        if data_normalization:
            y_train, y_test, y = self.normalize_data(
                y_train=y_train, y_test=y_test, y=y
            )

        output_device = self.output_device
        x_train, y_train = x_train.to(output_device), y_train.to(output_device)
        x_test, y_test = x_test.to(output_device), y_test.to(output_device)
        x, y = x.to(output_device), y.to(output_device)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        return x_train, y_train, x_test, y_test, x, y

    def train(
        self,
        model: gpytorch.models,
        likelihood: gpytorch.likelihoods,
        x_train: torch.tensor,
        y_train: torch.tensor,
        training_iters: int = 1000,
        learning_rate: float = 0.1,
        optimizer=torch.optim.Adam,
    ) -> None:
        """
        Train the model using the training data with the given
        parameters.

        Args:
            model (gpytorch.models): The model to train.
            likelihood (gpytorch.likelihoods): The likelihood function.
            x_train (torch.tensor): The training data.
            y_train (torch.tensor): The training labels.
            training_iters (int, optional): The number of training
            iterations. Defaults to 1000.
            learning_rate (float, optional): The learning rate. Defaults
            to 0.1.
            optimizer (_type_, optional): The optimizer. Defaults to
            torch.optim.Adam.
        """
        # Use the adam optimizer
        optimizer = optimizer(model.parameters(), lr=learning_rate)
        # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(training_iters):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(x_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.backward()
            print(
                "Iteration %d/%d - Loss: %.3f "
                % (i + 1, training_iters, loss.item())
            )
            optimizer.step()

    def predict(
        self,
        model: gpytorch.models,
        likelihood: gpytorch.likelihoods,
        x_test: torch.tensor,
        y_test: torch.tensor,
    ) -> gpytorch.distributions:
        """
        Predict the target values using the model and likelihood
        functions. Also, calculate the metrics.

        Args:
            model (gpytorch.models): The model to use for prediction.
            likelihood (gpytorch.likelihoods): The likelihood function.
            x_test (torch.tensor): The testing data.
            y_test (torch.tensor): The testing labels.

        Returns:
            gpytorch.distributions (torch.tensor): The predictions.
        """
        model.eval()
        likelihood.eval()

        predictions = None

        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Test points are regularly spaced along [0,1]
            predictions = likelihood(model(x_test))

        pred_mean = predictions.mean

        # Get the metrics
        mae = gpytorch.metrics.mean_absolute_error(predictions, y_test)
        mse = gpytorch.metrics.mean_squared_error(predictions, y_test)
        r2 = r2_score(y_test, pred_mean)
        mlss = gpytorch.metrics.mean_standardized_log_loss(predictions, y_test)
        nlpd = gpytorch.metrics.negative_log_predictive_density(
            predictions, y_test
        )

        print(f"Test MAE: {mae}")
        print(f"Test MSE: {mse}")
        print(f"Test R2: {r2}")
        print(f"Test MLSS: {mlss}")
        print(f"Test NLPD: {nlpd}")

        return predictions
