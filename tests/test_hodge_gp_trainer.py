import gpytorch
import numpy as np
import pytest
import torch
from torcheval.metrics.functional import r2_score

from sclibrary import load_dataset
from sclibrary.hogde_gp import ExactGPModel, HodgeGPTrainer
from sclibrary.hogde_gp.kernel_serializer import KernelSerializer


@pytest.fixture
def hodge_gp_trainer():
    sc, _, flow = load_dataset("forex")
    y = np.fromiter(flow.values(), dtype=float)

    B1 = sc.incidence_matrix(rank=1).toarray()
    B2 = sc.incidence_matrix(rank=2).toarray()
    L1 = sc.hodge_laplacian_matrix(rank=1).toarray()
    L1l = sc.lower_laplacian_matrix(rank=1).toarray()
    L1u = sc.upper_laplacian_matrix(rank=1).toarray()

    hogde_gp = HodgeGPTrainer(
        B1=B1,
        B2=B2,
        L1=L1,
        L1l=L1l,
        L1u=L1u,
        y=y,
    )

    yield hogde_gp


class TestHodgeGPTrainer:

    def test_normalize_data(self, hodge_gp_trainer):
        y_train = np.array([1, 2, 3, 4, 5])
        y_test = np.array([6, 7, 8, 9, 10])
        y = np.array([11, 12, 13, 14, 15])

        # covert to torch tensors
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        normalized_y_train, _, _ = hodge_gp_trainer.normalize_data(
            y_train, y_test, y
        )

        # Check if the mean of normalized_y_train is close to 0
        assert torch.isclose(torch.mean(normalized_y_train), torch.tensor(0.0))
        # Check if the standard deviation of normalized_y_train is close to 1
        assert torch.isclose(torch.std(normalized_y_train), torch.tensor(1.0))

    def test_get_laplacians(self, hodge_gp_trainer):
        laplacians = hodge_gp_trainer.get_laplacians()
        assert len(laplacians) == 3
        # check if all laplacians are torch tensors
        for laplacian in laplacians:
            assert type(laplacian) == torch.Tensor

    def test_get_incidence_matrices(self, hodge_gp_trainer):
        incidence_matrices = hodge_gp_trainer.get_incidence_matrices()
        assert len(incidence_matrices) == 2
        # check if all incidence matrices are torch tensors
        for incidence_matrix in incidence_matrices:
            assert type(incidence_matrix) == torch.Tensor

    def test_get_eigenvpairs(self, hodge_gp_trainer):
        eigenpairs = hodge_gp_trainer.get_eigenpairs()
        assert len(eigenpairs) == 6

        for eigenpair in eigenpairs:
            assert type(eigenpair) == torch.Tensor

    def test_train_test_split(self, hodge_gp_trainer):
        train_ratio = 0.2
        X_train, y_train, X_test, y_test, x, y = (
            hodge_gp_trainer.train_test_split(train_ratio=train_ratio)
        )

        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

        assert len(X_train) == int(len(x) * train_ratio)
        assert len(X_test) == len(x) - len(X_train)

    def test_trainer(self, hodge_gp_trainer):
        kernel_type = "matern"
        data_name = "forex"

        train_ratio = 0.2
        X_train, y_train, X_test, y_test, _, _ = (
            hodge_gp_trainer.train_test_split(train_ratio=train_ratio)
        )

        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        eigenpairs = hodge_gp_trainer.get_eigenpairs()
        kernel = KernelSerializer().serialize(
            eigenpairs=eigenpairs, kernel_type=kernel_type, data_name=data_name
        )

        model = ExactGPModel(
            X_train, y_train, likelihood, kernel, mean_function=None
        )

        model.train()
        likelihood.train()
        hodge_gp_trainer.train(model, likelihood, X_train, y_train)

        # test predictions
        predictions = hodge_gp_trainer.predict(
            model, likelihood, X_test, y_test
        )

        mae = gpytorch.metrics.mean_absolute_error(predictions, y_test)
        mse = gpytorch.metrics.mean_squared_error(predictions, y_test)
        pred_mean = predictions.mean
        r2 = r2_score(y_test, pred_mean)

        assert mae < 0.1
        assert mse < 0.01
        assert r2 > 0.99
