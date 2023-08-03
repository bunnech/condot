#!/usr/bin/python3

# imports
import anndata
import torch
from torch.utils.data import DataLoader


def transport(config, model, dataset, condition, return_as="anndata", dosage=None, **kwargs):
    name = config.model.get("name", "condot")
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    inputs = next(iter(loader))

    if name == "condot":
        outputs = transport_condot(model, (inputs, condition))

    elif name == "autoencoder":
        outputs = transport_autoencoder(
            model, (inputs, config.data.source, config.data.target), **kwargs)

    else:
        raise ValueError

    if dosage is not None:
        outputs = (1 - dosage) * inputs + dosage * outputs

    if return_as == "anndata":
        outputs = anndata.AnnData(
            outputs.detach().numpy(),
            obs=dataset.adata.obs.copy(),
            var=dataset.adata.var.copy(),
        )

    return outputs


def transport_condot(model, inputs):
    source, condition = inputs
    condition = condition[0] if isinstance(condition, list) else condition
    f, g = model
    g.eval()
    outputs = g.transport(source.requires_grad_(True), condition)
    return outputs


def transport_autoencoder(model, inputs, decode=True):
    inputs, source, target = inputs
    model.eval()
    shift = model.code_means[target] - model.code_means[source]
    codes = model.encode(inputs)
    if not decode:
        return codes + shift

    outputs = model.decode(codes + shift)
    return outputs
