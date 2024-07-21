import torch

from topomodelx.nn.simplicial.sccnn import Network as SCCNN


class Network(torch.nn.Module):
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
        super().__init__()
        self.base_model = SCCNN(
            in_channels_all=in_channels_all,
            hidden_channels_all=hidden_channels_all,
            conv_order=conv_order,
            sc_order=max_rank,
            update_func=update_func,
            n_layers=n_layers,
        )
        out_channels_0, _, _ = hidden_channels_all
        self.out_linear_0 = torch.nn.Linear(out_channels_0, out_channels)

    def forward(self, x_all, laplacian_all, incidence_all):
        x_all = self.base_model(x_all, laplacian_all, incidence_all)
        x_0, _, _ = x_all

        """
        We pass the output on the nodes to a linear layer and use that to
        generate a probability label for nodes.
        """
        x_0, _, _ = x_all
        logits = self.out_linear_0(x_0)

        return torch.sigmoid(logits)
