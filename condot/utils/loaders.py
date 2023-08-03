#!/usr/bin/python3

# internal imports
import condot.models
from condot.data.cell import load_cell_data


def load_data(config, **kwargs):
    data_type = config.get("data.type", "cell")
    if data_type in ["cell", "cell-merged"]:
        loadfxn = load_cell_data

    else:
        raise ValueError

    return loadfxn(config, **kwargs)


def load_model(config, restore=None, loader=None, **kwargs):
    name = config.get("model.name", "condot")
    if name == "condot":
        loadfxn = condot.models.load_condot_model

    elif name == "autoencoder":
        loadfxn = condot.models.load_autoencoder_model

    else:
        raise ValueError

    return loadfxn(config, restore=restore, loader=loader, **kwargs)


def load(config, restore=None, include_model_kwargs=False, **kwargs):
    loader, model_kwargs = load_data(config, include_model_kwargs=True, **kwargs)
    model, opt = load_model(config, restore=restore, loader=loader, **model_kwargs)

    if include_model_kwargs:
        return model, opt, loader, model_kwargs

    return model, opt, loader
