#!/usr/bin/python3

# imports
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# internal imports
from condot.networks.ae import AutoEncoder


def load_optimizer(config, params):
    kwargs = dict(config.get("optim", {}))
    assert kwargs.pop("optimizer", "Adam") == "Adam"
    optim = torch.optim.Adam(params, **kwargs)
    return optim


def load_networks(config, **kwargs):
    kwargs = kwargs.copy()
    kwargs.update(dict(config.get("model", {})))
    name = kwargs.pop("name")
    if "dimension_reduction" in config.data:
        kwargs.update({"latent_dim": 50})

    if name == "autoencoder":
        model = AutoEncoder

    else:
        raise ValueError

    return model(**kwargs)


def load_autoencoder_model(config, restore=None, **kwargs):
    model = load_networks(config, **kwargs)
    optim = load_optimizer(config, model.parameters())

    if restore is not None and Path(restore).exists():
        ckpt = torch.load(restore)
        model.load_state_dict(ckpt["model_state"])
        optim.load_state_dict(ckpt["optim_state"])
        if config.model.name == "autoencoder" and "code_means" in ckpt:
            model.code_means = ckpt["code_means"]

    # push to GPU
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return model, optim


def compute_autoencoder_shift(model, dataset, labels):
    model.code_means = dict()

    inputs = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False)))
    codes = model.encode(inputs)

    for key in labels.unique():
        mask = labels == key
        model.code_means[key] = codes[mask.values].mean(0)

    return model.code_means
