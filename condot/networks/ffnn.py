#!/usr/bin/python3

# imports
import torch
from torch import autograd
from torch import nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
}


class FFNN(nn.Module):

    def __init__(
        self,
        input_dim,
        input_dim_label,
        hidden_units,
        activation="leakyrelu",
        combinator=False,
        embedding=False,
        **kwargs
    ):
        super(FFNN, self).__init__()

        if isinstance(activation, str):
            activation = ACTIVATIONS[activation.lower().replace("_", "")]
        self.sigma = activation
        self.softplus_wz_kernels = True

        self.combinator = combinator
        self.embedding = embedding

        units = hidden_units
        self.n_layers = len(units)
        self.input_dim = input_dim
        self.input_dim_label = input_dim_label

        self.wxs = nn.ModuleList(
            [nn.Linear(idim, odim, bias=True) for idim, odim in zip(
                [input_dim] + [2 * units[0]] + units[1:], units + [1])]
        )

        self.wy = nn.Linear(input_dim_label, units[0], bias=True)

    def forward(self, x, y):
        # get label (combination) embedding
        if self.combinator:
            y = self.combinator(y)
        elif self.embedding and not self.combinator:
            y = self.embedding.forward(y)
        elif not self.embedding and not self.combinator:
            y = y
        else:
            raise ValueError

        if y.ndim == 1:
            y = y.repeat(x.size(dim=0), 1)

        z = torch.cat((self.wxs[0](x), self.wy(y)), dim=1)

        for i in range(1, self.n_layers + 1):
            z = self.wxs[i](z)

            if i != self.n_layers:
                z = self.sigma(0.2)(z)
        return z

    def transport(self, x, y):
        assert x.requires_grad

        (output,) = autograd.grad(
            self.forward(x, y),
            x,
            create_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((x.size()[0], 1), device=x.device).float(),
        )
        return output
