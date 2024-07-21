import torch

from topomodelx.nn.simplicial.scnn import SCNN


class Network(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        conv_order_down,
        conv_order_up,
        n_layers=2,
    ):
        super().__init__()
        self.base_model = SCNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            conv_order_down=conv_order_down,
            conv_order_up=conv_order_up,
            n_layers=n_layers,
        )
        self.linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, laplacian_down, laplacian_up):
        x = self.base_model(x, laplacian_down, laplacian_up)
        return self.linear(x)
