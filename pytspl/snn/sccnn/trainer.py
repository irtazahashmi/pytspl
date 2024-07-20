import numpy as np
import torch

from .network import Network


class SCCNNTrainer:

    def __init__(
        self,
        in_channels_all,
        hidden_channels_all,
        out_channels,
        conv_order,
        max_rank,
        update_func=None,
        n_layers=2,
    ):

        self.in_channels_all = in_channels_all
        self.hidden_channels_all = hidden_channels_all
        self.out_channels = out_channels
        self.conv_order = conv_order
        self.max_rank = max_rank
        self.update_func = update_func
        self.n_layers = n_layers

        self._init_network()

    def _init_network(self):
        self.network = Network(
            in_channels_all=self.in_channels_all,
            hidden_channels_all=self.hidden_channels_all,
            out_channels=self.out_channels,
            conv_order=self.conv_order,
            max_rank=self.max_rank,
            update_func=self.update_func,
            n_layers=self.n_layers,
        )
        self.parameters = self.network.parameters()

    def train(
        self,
        feats: torch.tensor,
        incidence_mats: torch.tensor,
        laplacians: torch.tensor,
        y_train: torch.tensor,
        optimizer: torch.optim,
        num_epochs: int,
    ):
        for epoch_i in range(1, num_epochs + 1):
            epoch_loss = []
            self.network.train()
            optimizer.zero_grad()
            y_hat = self.network(feats, laplacians, incidence_mats)
            y_hat = torch.softmax(y_hat, dim=1)
            loss = torch.nn.functional.binary_cross_entropy(
                y_hat[: len(y_train)].float(), y_train.float()
            )
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            y_pred = torch.where(y_hat > 0.5, torch.tensor(1), torch.tensor(0))
            accuracy = (
                (y_pred[: len(y_train)] == y_train)
                .all(dim=1)
                .float()
                .mean()
                .item()
            )

            print(
                f"Epoch: {epoch_i} loss: {np.mean(epoch_loss):.4f} "
                + f"Train_acc: {accuracy:.4f}",
                flush=True,
            )

    def predict(
        self,
        feats: torch.tensor,
        incidence_mats: torch.tensor,
        laplacians: torch.tensor,
        y_test: torch.tensor,
    ):
        with torch.no_grad():
            y_hat_test = self.network(feats, laplacians, incidence_mats)
            # Projection to node-level
            y_hat_test = torch.softmax(y_hat_test, dim=1)
            y_pred_test = torch.where(
                y_hat_test > 0.5, torch.tensor(1), torch.tensor(0)
            )
            test_accuracy = (
                torch.eq(y_pred_test[-len(y_test) :], y_test)
                .all(dim=1)
                .float()
                .mean()
                .item()
            )
            print(f"Test_acc: {test_accuracy:.4f}", flush=True)
