#!/usr/bin/python3

# imports
import torch
from torch import nn


class NonNegativeLinear(nn.Linear):
    def __init__(self, *args, beta=1.0, **kwargs):
        super(NonNegativeLinear, self).__init__(*args, **kwargs)
        self.beta = beta

    def forward(self, x):
        return nn.functional.linear(x, self.kernel(), self.bias)

    def kernel(self):
        return nn.functional.softplus(self.weight, beta=self.beta)


class PosDefDense(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(PosDefDense, self).__init__(*args, **kwargs)

    def forward(self, x):
        kernel = nn.functional.linear(x, self.weight, None)
        return nn.functional.linear(kernel, torch.transpose(self.weight, 0, 1), self.bias)


class PosDefPotentials(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(PosDefPotentials, self).__init__(*args, **kwargs)

        self.weight = nn.parameter.Parameter(torch.empty((
            self.out_features, self.in_features, self.in_features)))
        if self.bias is not None:
            self.bias = nn.parameter.Parameter(
                torch.empty(self.out_features, self.in_features))

    def forward(self, x):
        if self.bias is not None:
            out = torch.reshape(x, (-1, x.shape[-1])) if x.ndim == 1 else x
            out = out[..., None] - self.bias.T[None, ...]
            out = torch.einsum('ikb,bjk->ijb', out, self.weight)
            out = 0.5 * out * out
            return torch.sum(out, dim=1)
        else:
            # TODO: implement!
            return x
