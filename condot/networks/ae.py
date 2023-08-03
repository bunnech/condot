#!/usr/bin/python3

# imports
from collections import namedtuple
import torch
from torch import nn


def dnn(
    dinput,
    doutput,
    hidden_units=(16, 16),
    activation="ReLU",
    dropout=0.0,
    batch_norm=False,
    net_fn=nn.Sequential,
    **kwargs
):

    if isinstance(hidden_units, int):
        hidden_units = [hidden_units]
    hidden_units = list(hidden_units)

    layer_sizes = zip([dinput] + hidden_units[:-1], hidden_units)

    if isinstance(activation, str):
        Activation = getattr(nn, activation)
    else:
        Activation = activation

    layers = list()
    for indim, outdim in layer_sizes:
        layers.append(nn.Linear(indim, outdim, **kwargs))

        if batch_norm:
            layers.append(nn.BatchNorm1d(outdim))

        layers.append(Activation())

        if dropout is not None and dropout > 0:
            layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(hidden_units[-1], doutput))
    net = nn.Sequential(*layers)
    return net


class AutoEncoder(nn.Module):
    LossComps = namedtuple("AELoss", "mse reg")
    Outputs = namedtuple("AEOutputs", "recon code")

    def __init__(
        self,
        input_dim,
        latent_dim,
        encoder_net=None,
        decoder_net=None,
        hidden_units=None,
        beta=0,
        dropout=0,
        mse=None,
        **kwargs
    ):

        super(AutoEncoder, self).__init__()

        if encoder_net is None:
            assert hidden_units is not None
            encoder_net = self.build_encoder(
                input_dim, latent_dim, hidden_units, dropout=dropout
            )

        if decoder_net is None:
            assert hidden_units is not None
            decoder_net = self.build_decoder(
                input_dim, latent_dim, hidden_units, dropout=dropout
            )

        self.beta = beta
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net

        if mse is None:
            mse = nn.MSELoss(reduction="none")

        self.mse = mse

        return

    def build_encoder(self, input_dim, latent_dim, hidden_units, **kwargs):
        net = dnn(
            dinput=input_dim, doutput=latent_dim, hidden_units=hidden_units, **kwargs
        )
        return net

    def build_decoder(self, input_dim, latent_dim, hidden_units, **kwargs):
        net = dnn(
            dinput=latent_dim,
            doutput=input_dim,
            hidden_units=hidden_units[::-1],
            **kwargs
        )
        return net

    def encode(self, inputs, **kwargs):
        return self.encoder_net(inputs, **kwargs)

    def decode(self, code, **kwargs):
        return self.decoder_net(code, **kwargs)

    def outputs(self, inputs, **kwargs):
        code = self.encode(inputs, **kwargs)
        recon = self.decode(code, **kwargs)
        outputs = self.Outputs(recon, code)
        return outputs

    def loss(self, inputs, outputs):
        mse = self.mse(outputs.recon, inputs).mean(dim=-1)
        reg = torch.norm(outputs.code, dim=-1) ** 2
        total = mse + self.beta * reg
        comps = self.LossComps(mse, reg)
        return total, comps

    def forward(self, inputs, **kwargs):
        outs = self.outputs(inputs, **kwargs)
        loss, comps = self.loss(inputs, outs)

        return loss, comps, outs
