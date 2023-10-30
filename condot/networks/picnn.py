#!/usr/bin/python3

# imports
import torch
from torch import autograd
from torch import nn

# internal imports
from condot.networks.layers import NonNegativeLinear
from condot.utils.helpers import to_device


ACTIVATIONS = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
}


class PICNN(nn.Module):
    def __init__(
        self,
        input_dim,
        input_dim_label,
        hidden_units,
        activation="leakyrelu",
        softplus_wz_kernels=False,
        softplus_beta=1,
        fnorm_penalty=0,
        kernel_init_fxn=None,
        combinator=False,
        embedding=False,
        neural_embedding=None,
        **kwargs
    ):

        super(PICNN, self).__init__()
        self.fnorm_penalty = fnorm_penalty
        self.softplus_wz_kernels = softplus_wz_kernels

        if isinstance(activation, str):
            activation = ACTIVATIONS[activation.lower().replace("_", "")]
        self.sigma = activation

        self.combinator = combinator
        self.embedding = embedding

        units = hidden_units
        self.n_layers = len(units)
        self.input_dim = input_dim
        self.input_dim_label = input_dim_label

        if self.softplus_wz_kernels:
            def Linear(*args, **kwargs):
                return NonNegativeLinear(*args, **kwargs, beta=softplus_beta)

        else:
            Linear = nn.Linear

        self.w = nn.ModuleList(
            [nn.Linear(idim, odim, bias=True) for idim, odim in zip(
                [input_dim_label] + [units[0]] + units[:-2], [units[0]] + units[:-1])]
        )

        self.wz = nn.ModuleList(
            [Linear(idim, odim, bias=False) for idim, odim in zip(
                units, units[1:] + [1])]
        )

        self.wzu = nn.ModuleList(
            [nn.Linear(idim, odim, bias=True) for idim, odim in zip(
                units, units)]
        )

        self.wx = nn.ModuleList(
            [nn.Linear(input_dim, odim, bias=True) for odim in (
                units + [1])]
        )

        self.wxu = nn.ModuleList(
            [nn.Linear(idim, input_dim, bias=True) for idim in (
                [units[0]] + units)]
        )

        self.wu = nn.ModuleList(
            [nn.Linear(idim, odim, bias=False) for idim, odim in zip(
                [units[0]] + units, units + [1])]
        )

        # neural embedding encoding
        self.neural_embedding = neural_embedding
        if self.neural_embedding is not None:
            self.we = nn.Linear(in_features=self.neural_embedding[0],
                                out_features=self.neural_embedding[1])

        if kernel_init_fxn is not None:
            for layer in self.wx:
                kernel_init_fxn(layer.weight)
                nn.init.zeros_(layer.bias)

            for layer in self.wz:
                kernel_init_fxn(layer.weight)

            for layer in self.w:
                kernel_init_fxn(layer.weight)
                nn.init.zeros_(layer.bias)

            for layer in self.wu:
                kernel_init_fxn(layer.weight)

            for layer in self.wxu:
                kernel_init_fxn(layer.weight)
                nn.init.zeros_(layer.bias)

            for layer in self.wzu:
                kernel_init_fxn(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, y):

        x = to_device(x)
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

        if self.neural_embedding:
            y = nn.Sigmoid()(self.we(y))

        u = y

        for i in range(self.n_layers):
            u = self.sigma(0.2)(self.w[i](u))
            if i == 0:
                z = self.sigma(0.2)(self.wx[i](torch.mul(x, self.wxu[i](u))) + self.wu[i](u))
                z = z * z
            else:
                z = self.sigma(0.2)(
                    self.wz[i - 1](torch.mul(z, nn.functional.softplus(self.wzu[i - 1](u))))
                    + self.wx[i](torch.mul(x, self.wxu[i](u))) + self.wu[i](u)
                  )

        z = (self.wz[-1](torch.mul(z, nn.functional.softplus(self.wzu[-1](u))))
             + self.wx[-1](torch.mul(x, self.wxu[-1](u))) + self.wu[-1](u))
        return z

    def transport(self, x, y):
        x = to_device(x)
        assert x.requires_grad

        (output,) = autograd.grad(
            self.forward(x, y),
            to_device(x),
            create_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((to_device(x).size()[0], 1), device=x.device).float(),
        )
        return output

    def clamp_w(self):
        if self.softplus_wz_kernels:
            return

        for w in self.wz:
            w.weight.data = w.weight.data.clamp(min=0)
        return

    def penalize_w(self):
        return self.fnorm_penalty * sum(
            map(lambda x: torch.nn.functional.relu(-x.weight).norm(), self.wz)
        )
