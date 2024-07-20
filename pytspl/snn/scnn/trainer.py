import numpy as np
import torch

from .network import Network


class SCNNTrainer:

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        conv_order_down,
        conv_order_up,
        n_layers,
    ):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.conv_order_down = conv_order_down
        self.conv_order_up = conv_order_up
        self.n_layers = n_layers

        self._init_network()

    def _init_network(self):
        self.network = Network(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            conv_order_down=self.conv_order_down,
            conv_order_up=self.conv_order_up,
            n_layers=self.n_layers,
        )
        self.parameters = self.network.parameters()

    def train(
        self,
        feats: torch.tensor,
        incidence_1: torch.tensor,
        laplacian_down: torch.tensor,
        laplacian_up: torch.tensor,
        y_train: torch.tensor,
        optimizer: torch.optim,
        num_epochs: int,
    ):
        for epoch_i in range(1, num_epochs + 1):
            epoch_loss = []
            self.network.train()
            optimizer.zero_grad()

            y_hat_edge = self.network(feats, laplacian_down, laplacian_up)
            y_hat = torch.softmax(
                torch.sparse.mm(incidence_1, y_hat_edge), dim=1
            )
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
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
        incidence_1: torch.tensor,
        laplacian_down: torch.tensor,
        laplacian_up: torch.tensor,
        y_test: torch.tensor,
    ):
        with torch.no_grad():
            y_hat_edge_test = self.network(feats, laplacian_down, laplacian_up)
            # Projection to node-level
            y_hat_test = torch.softmax(
                torch.sparse.mm(incidence_1, y_hat_edge_test), dim=1
            )
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
