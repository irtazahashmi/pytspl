"""Module for training the Hodge Gaussian Process model."""

import gpytorch
import numpy as np
import torch
from torcheval.metrics.functional import r2_score

DATA_TYPE = torch.float32


class HodgeGPTrainer:
    def __init__(
        self,
        B1: np.ndarray,
        B2: np.ndarray,
        L1: np.ndarray,
        L1l: np.ndarray,
        L1u: np.ndarray,
        y: np.ndarray,
        output_device: str = "cpu",
    ):
        """
        Initialize the HodgeGPTrainer class.

        Args:
            B1 (np.ndarray): The incidence matrix B1.
            B2 (np.ndarray): The incidence matrix B2.
            L1 (np.ndarray): The Laplacian matrix L1.
            L1l (np.ndarray): The lower Laplacian matrix L1l.
            L1u (np.ndarray): The upper Laplacian matrix L1u.
            y (np.ndarray): The target values.
            output_device (str, optional): The output device for
            the tensors. Defaults to "cpu".
        """

        self.B1 = B1
        self.B2 = B2
        self.L1 = L1
        self.L1l = L1l
        self.L1u = L1u
        self.y = y

        self.output_device = torch.device(output_device)

    def normalize_data(
        self, y_train: np.ndarray, y_test: np.ndarray, y: np.ndarray
    ) -> tuple:
        """
        Normalize the target values.

        Args:
            y_train (np.ndarray): The training target values.
            y_test (np.ndarray): The testing target values.
            y (np.ndarray): The target values.

        Returns:
            tuple(np.ndarray, np.ndarray, np.ndarray): The normalized
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
        np.random.seed(seed)

        y = self.y

        n1 = self.B1.shape[1]
        num_train = int(train_ratio * n1)

        x = np.arange(n1)
        random_perm = np.random.permutation(x)
        train_ids, test_ids = random_perm[:num_train], random_perm[num_train:]

        x_train, x_test = x[train_ids], x[test_ids]
        y_train, y_test = y[train_ids], y[test_ids]

        # print shapes
        print(f"x_train: {x_train.shape}")
        print(f"x_test: {x_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test: {y_test.shape}")

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
            y_train, y_test, y = self.normalize_data(y_train, y_test, y)

        output_device = self.output_device
        x_train, y_train = x_train.to(output_device), y_train.to(output_device)
        x_test, y_test = x_test.to(output_device), y_test.to(output_device)
        x, y = x.to(output_device), y.to(output_device)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        return x_train, y_train, x_test, y_test, x, y

    def get_laplacians(self) -> list:
        """
        Return the Laplacian matrices as a list of tensors.

        Returns:
            list(torch.tensor): The Laplacian matrices.
        """
        print("L1: ", self.L1.shape)
        print("L1l: ", self.L1l.shape)
        print("L1u: ", self.L1u.shape)

        L1 = torch.tensor(self.L1, dtype=DATA_TYPE)
        L1_down = torch.tensor(self.L1l, dtype=DATA_TYPE)
        L1_up = torch.tensor(self.L1u, dtype=DATA_TYPE)

        laplacians = [L1, L1_down, L1_up]
        laplacians = [
            laplacian.to(self.output_device) for laplacian in laplacians
        ]

        return laplacians

    def get_incidence_mats(self) -> list:
        """
        Return the incidence matrices as a list of tensors.

        Returns:
            list(torch.tensor): The incidence matrices.
        """
        print("B1: ", self.B1.shape)
        print("B2: ", self.B2.shape)

        B1 = torch.tensor(self.B1, dtype=DATA_TYPE)
        B2 = torch.tensor(self.B2, dtype=DATA_TYPE)

        output_device = self.output_device
        incidence_matrices = [B1.to(output_device), B2.to(output_device)]

        return incidence_matrices

    def get_eigenpairs(self) -> list:
        """
        Return the eigenpairs of the Laplacian matrices.

        Returns:
            list(torch.tensor): The eigenpairs of the Laplacian matrices.
        """
        eigvals, eigvecs = np.linalg.eigh(self.L1)

        total_var = np.diag(eigvecs.T @ self.L1 @ eigvecs)
        total_div = np.diag(eigvecs.T @ self.L1l @ eigvecs)
        total_curl = np.diag(eigvecs.T @ self.L1u @ eigvecs)

        harm_eflow = np.where(np.array(total_var) <= 1e-4)[0]
        grad_eflow = np.where(np.array(total_div) > 1e-4)[0]
        curl_eflow = np.where(np.array(total_curl) >= 1e-3)[0]

        harm_evectors = torch.tensor(eigvecs[:, harm_eflow], dtype=DATA_TYPE)
        grad_evectors = torch.tensor(eigvecs[:, grad_eflow], dtype=DATA_TYPE)
        curl_evectors = torch.tensor(eigvecs[:, curl_eflow], dtype=DATA_TYPE)
        harm_evalues = torch.tensor(eigvals[harm_eflow], dtype=DATA_TYPE)
        grad_evalues = torch.tensor(eigvals[grad_eflow], dtype=DATA_TYPE)
        curl_evalues = torch.tensor(eigvals[curl_eflow], dtype=DATA_TYPE)

        output_device = self.output_device

        harm_evectors = harm_evectors.to(output_device)
        grad_evectors = grad_evectors.to(output_device)
        curl_evectors = curl_evectors.to(output_device)
        harm_evalues = harm_evalues.to(output_device)
        grad_evalues = grad_evalues.to(output_device)
        curl_evalues = curl_evalues.to(output_device)

        eigenpairs = [
            harm_evectors,
            grad_evectors,
            curl_evectors,
            harm_evalues,
            grad_evalues,
            curl_evalues,
        ]

        return eigenpairs

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
                "Iter %d/%d - Loss: %.3f "
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
