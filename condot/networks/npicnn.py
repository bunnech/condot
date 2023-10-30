#!/usr/bin/python3

# imports
import numpy as np
import torch
from torch import autograd
from torch import nn
import scipy as sc
from functools import reduce

# internal imports
from condot.networks.layers import NonNegativeLinear, PosDefPotentials
from condot.utils.helpers import to_device

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
}


class NPICNN(nn.Module):
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
        init_type="identity",
        init_inputs=None,
        num_labels=None,
        name=None,
        **kwargs
    ):

        super(NPICNN, self).__init__()
        self.fnorm_penalty = fnorm_penalty
        self.softplus_wz_kernels = softplus_wz_kernels

        if isinstance(activation, str):
            activation = ACTIVATIONS[activation.lower().replace("_", "")]
        self.sigma = activation

        self.combinator = combinator
        self.embedding = embedding
        self.use_init = False

        # check if labelled Gaussian maps were provided
        if init_type == "identity":
            labels, factors, means = self.compute_identity_maps(
                init_inputs, input_dim)
        elif init_type == "gaussian":
            if isinstance(init_inputs, tuple):
                labels, factors, means = init_inputs
            else:
                labels, factors, means = self.compute_gaussian_maps(
                    init_inputs, name)
        else:
            raise NotImplementedError

        units = [num_labels] + hidden_units

        self.n_layers = len(units)
        self.input_dim = input_dim
        self.input_dim_label = input_dim_label

        if self.softplus_wz_kernels:
            def Linear(*args, **kwargs):
                return NonNegativeLinear(*args, **kwargs, beta=softplus_beta)
            # this function should be inverse map of function used in PositiveDense layers
            rescale = lambda x: np.log(np.exp(x) - 1)
        else:
            Linear = nn.Linear
            rescale = lambda x: x

        # self layers for hidden state u, when contributing all ~0
        self.w = list()
        for idim, odim in zip([self.input_dim_label] + [units[0]] + units[:-2],
                              [units[0]] + units[:-1]):
            _w = nn.Linear(idim, odim, bias=True)
            kernel_init_fxn(_w.weight)
            self.w.append(_w)
        self.w = nn.ModuleList(self.w)

        # first layer for hidden state performs a comparison with database
        self.w_0 = nn.Linear(labels.shape[1], num_labels, bias=False)
        if labels.shape[0] == num_labels:
            with torch.no_grad():
                self.w_0.weight.copy_(labels)

        # auto layers for z, should be mean operators with no bias
        # keep track of previous size to normalize accordingly
        normalization = 1
        self.wz = list()
        for idim, odim in zip(units, units[1:] + [1]):
            _wz = Linear(idim, odim, bias=False)
            nn.init.constant_(_wz.weight, rescale(1.0 / normalization))

            self.wz.append(_wz)
            normalization = odim
        self.wz = nn.ModuleList(self.wz)

        # for family of convex functions stored in z, if using init then first
        # vector z_0 has as many values as # of convex potentials.
        self.w_z0 = PosDefPotentials(self.input_dim, num_labels, bias=True)
        if labels.shape[0] == num_labels:
            with torch.no_grad():
                self.w_z0.weight.copy_(factors)
                self.w_z0.bias.copy_(means)

        # cross layers for convex functions z / hidden state u
        # initialized to be identity first with 0 bias
        # and then ~0 + 1 bias to ensure identity
        self.wzu = list()

        _wzu = nn.Linear(num_labels, units[0], bias=True)
        kernel_init_fxn(_wzu.weight)
        nn.init.constant_(_wzu.bias, rescale(1.0))
        self.wzu.append(_wzu)
        zu_layers = zip(units[1:], units[1:])

        for idim, odim in zu_layers:
            _wzu = nn.Linear(idim, odim, bias=True)
            kernel_init_fxn(_wzu.weight)
            nn.init.constant_(_wzu.bias, rescale(1.0))
            self.wzu.append(_wzu)
        self.wzu = nn.ModuleList(self.wzu)

        # self layers for x, ~0
        self.wx = list()
        for odim in (units + [1]):
            _wx = nn.Linear(input_dim, odim, bias=True)
            kernel_init_fxn(_wx.weight)
            nn.init.zeros_(_wx.bias)
            self.wx.append(_wx)
        self.wx = nn.ModuleList(self.wx)

        # cross layers for x / hidden state u, all ~0
        self.wxu = list()
        for idim in ([units[0]] + units):
            _wxu = nn.Linear(idim, input_dim, bias=True)
            kernel_init_fxn(_wxu.weight)
            nn.init.zeros_(_wxu.bias)
            self.wxu.append(_wxu)
        self.wxu = nn.ModuleList(self.wxu)

        # self layers for hidden state u, to update z, all ~0
        self.wu = list()
        for idim, odim in zip([units[0]] + units, units + [1]):
            _wu = nn.Linear(idim, odim, bias=False)
            kernel_init_fxn(_wu.weight)
            self.wu.append(_wu)
        self.wu = nn.ModuleList(self.wu)

        # neural embedding encoding
        self.neural_embedding = neural_embedding
        if self.neural_embedding is not None:
            self.we = nn.Linear(in_features=self.neural_embedding[0],
                                out_features=self.neural_embedding[1])

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

        # initialize u and z
        u = y

        u = nn.functional.softmax(10 * self.w_0(u), dim=1)
        z_0 = self.w_z0(x)
        z = self.sigma(0.2)(z_0 * u)
        # apply k layers - 1
        for i in range(1, self.n_layers):
            u = self.sigma(0.2)(self.w[i](u))
            t_u = nn.functional.softplus(self.wzu[i - 1](u))
            z = self.sigma(0.2)(
                self.wz[i - 1](torch.mul(z, t_u))
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
            x,
            create_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((x.size()[0], 1), device=x.device).float(),
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

    def compute_gaussian_maps(self, dataloader, name):
        def dots(*args):
            return reduce(np.dot, args)

        def compute_moments(x, reg=1e-2, sqrt_inv=False):
            shape = x.size()
            z = x.reshape(shape[0], -1)
            mu = z.mean(dim=0).unsqueeze(0)
            z = (z - mu)
            sigma = torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
            # unbiased estimate
            sigma = sigma.sum(dim=0) / (shape[0] - 1)
            # regularize
            sigma = sigma + reg * np.eye(shape[1])

            if sqrt_inv:
                sigma_sqrt = sc.linalg.sqrtm(sigma)
                sigma_inv_sqrt = sc.linalg.inv(sigma_sqrt)
                return sigma, sigma_sqrt, sigma_inv_sqrt, mu
            else:
                return sigma, mu

        source_labels = list(dataloader.source.keys())
        target_labels = list(dataloader.target.keys())
        As = list()
        bs = list()

        for label in target_labels:
            if label in source_labels:
                source = torch.cat(
                    [next(iter(dataloader.source[label])) for _ in range(5)],
                    dim=0)
            else:
                assert len(source_labels) == 1
                source = torch.cat(
                    [next(iter(dataloader.source[source_labels[0]])) for _ in range(5)],
                    dim=0)
            target = torch.cat(
                [next(iter(dataloader.target[label])) for _ in range(5)],
                dim=0)

            if name == "f":
                source, target = target, source

            _, covs_sqrt, covs_inv_sqrt, mus = compute_moments(source, sqrt_inv=True)
            covt, mut = compute_moments(target, sqrt_inv=False)

            mo = sc.linalg.sqrtm(dots(covs_sqrt, covt.numpy(), covs_sqrt))
            A = dots(covs_inv_sqrt, mo, covs_inv_sqrt)
            b = mus.squeeze() - np.linalg.solve(A, mut.squeeze())

            bs.append(b)
            As.append(torch.from_numpy(sc.linalg.sqrtm(A)))

        if self.combinator:
            labels = torch.stack(
              [self.combinator.forward(label).squeeze() for label in target_labels])
        elif self.embedding and not self.combinator:
            labels = torch.stack([self.embedding.forward(label) for label in target_labels])

        return labels, torch.stack(As), torch.stack(bs)

    def compute_identity_maps(self, labels, input_dim):
        A = torch.eye(input_dim).reshape((1, input_dim, input_dim))
        As = A.repeat(len(labels), 1, 1)

        b = torch.zeros((1, input_dim))
        bs = b.repeat(len(labels), 1)

        if self.combinator:
            labels = torch.stack(
                [self.combinator.forward(label).squeeze() for label in labels])
        elif self.embedding and not self.combinator:
            labels = torch.stack(
                [self.embedding.forward(label) for label in labels])

        return labels, As, bs
