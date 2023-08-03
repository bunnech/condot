#!/usr/bin/python3

# imports
from pathlib import Path
from collections import namedtuple
import torch

# internal imports
from condot.networks.picnn import PICNN
from condot.networks.npicnn import NPICNN
from condot.networks import embedding
from condot.networks import combinator


FGPair = namedtuple("FGPair", "f g")


def load_networks(config, loader, **kwargs):
    def unpack_kernel_init_fxn(name="uniform", **kwargs):
        if name == "normal":

            def init(*args):
                return torch.nn.init.normal_(*args, **kwargs)

        elif name == "uniform":

            def init(*args):
                return torch.nn.init.uniform_(*args, **kwargs)

        else:
            raise ValueError

        return init

    # load embedding and combinator
    emb = load_embedding(config, loader)

    kwargs.setdefault("hidden_units", [64] * 4)
    kwargs.update(dict(config.get("model", {})))
    if config.model.embedding.type == "onehot":
        input_dim_label = len(emb.lb.classes_)
        kwargs.update({"input_dim_label": input_dim_label})

    elif config.model.embedding.type == "smacof":
        kwargs.update({"input_dim_label": config.model.embedding.dim})

    elif config.model.embedding.type == "fingerprint":
        kwargs.update({"neural_embedding": (2048, 10)})
        kwargs.update({"input_dim_label": 10})

    elif config.model.embedding.type == "value":
        kwargs.update({"input_dim_label": 1})

    if "input_dim_label" not in kwargs:
        kwargs.setdefault("input_dim_label", 2)

    # parameters specific to g are stored in config.model.g
    kwargs.pop("name")
    if "latent_dim" in kwargs:
        kwargs.pop("latent_dim")
    fupd = kwargs.pop("f", {})
    gupd = kwargs.pop("g", {})

    fkwargs = kwargs.copy()
    fkwargs.update(fupd)
    fkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(
        **fkwargs.pop("kernel_init_fxn")
    )

    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    gkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(
        **gkwargs.pop("kernel_init_fxn")
    )

    # if onehot embedding, do not use combinator module
    if config.model.embedding.type == "onehot":
        config.model.combinator = False

    # either load combinator with integrated embedding or embedding only
    if "combinator" in config.model and config.model.combinator:
        com = load_combinator(config, emb)
        gkwargs["combinator"] = com
        fkwargs["combinator"] = com
    else:
        gkwargs["embedding"] = emb
        fkwargs["embedding"] = emb

    if "init" in config.model:
        gkwargs["init_type"] = config.model.init
        fkwargs["init_type"] = config.model.init

        if config.model.init == "identity":
            if "num_labels" not in config.model:
                config.model.num_labels = len(loader.train.target.keys())

            gkwargs["num_labels"] = config.model.num_labels
            fkwargs["num_labels"] = config.model.num_labels
            gkwargs["init_inputs"] = loader.train.target.keys()
            fkwargs["init_inputs"] = loader.train.target.keys()

        elif "init" in config.model and config.model.init == "gaussian":
            if "num_labels" not in config.model:
                config.model.num_labels = len(loader.train.target.keys())
            gkwargs["num_labels"] = config.model.num_labels
            fkwargs["num_labels"] = config.model.num_labels
            gkwargs["init_inputs"] = loader.train
            fkwargs["init_inputs"] = loader.train
            gkwargs["name"] = "g"
            fkwargs["name"] = "f"

        else:
            raise ValueError()

        f = NPICNN(**fkwargs)
        g = NPICNN(**gkwargs)

    else:
        f = PICNN(**fkwargs)
        g = PICNN(**gkwargs)

    return f, g


def load_opts(config, f, g):
    kwargs = dict(config.get("optim", {}))
    assert kwargs.pop("optimizer", "Adam") == "Adam"

    fupd = kwargs.pop("f", {})
    gupd = kwargs.pop("g", {})

    fkwargs = kwargs.copy()
    fkwargs.update(fupd)
    fkwargs["betas"] = (fkwargs.pop("beta1", 0.9), fkwargs.pop("beta2", 0.999))

    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    gkwargs["betas"] = (gkwargs.pop("beta1", 0.9), gkwargs.pop("beta2", 0.999))

    opts = FGPair(
        f=torch.optim.Adam(f.parameters(), **fkwargs),
        g=torch.optim.Adam(g.parameters(), **gkwargs),
    )

    return opts


def load_embedding(config, loader, **kwargs):
    emb_type = config.get("model.embedding.type", "smacof")

    if emb_type == "smacof":
        model_embedding = embedding.EmbeddingSMACOF(
            config.model.embedding)
    elif emb_type == "fingerprint":
        model_embedding = embedding.EmbeddingFingerprint(
            config.model.embedding)
    elif emb_type == "onehot":
        model_embedding = embedding.EmbeddingOneHot(
            loader.train.target.keys())
    elif emb_type == "value":
        model_embedding = embedding.EmbeddingValue(
            config.model.embedding.factor)
    else:
        raise ValueError
    return model_embedding


def load_combinator(config, model_embedding, **kwargs):
    comb_type = config.get("model.combinator.type", "deepset")

    if comb_type == "deepset":
        model_combinator = combinator.DeepSet(
            input_dim=config.model.embedding.dim,
            hidden_units=config.model.combinator.hidden_units,
            pool=config.model.combinator.pool)
    else:
        raise ValueError
    return combinator.Combinator(model_combinator, model_embedding)


def load_condot_model(config, restore=None, loader=None, **kwargs):
    f, g = load_networks(config, loader, **kwargs)
    opts = load_opts(config, f, g)

    if restore is not None and Path(restore).exists():
        ckpt = torch.load(restore)
        f.load_state_dict(ckpt["f_state"])
        opts.f.load_state_dict(ckpt["opt_f_state"])

        g.load_state_dict(ckpt["g_state"])
        opts.g.load_state_dict(ckpt["opt_g_state"])

    return (f, g), opts


def compute_loss_g(f, g, source, condition, transport=None):
    if transport is None:
        transport = g.transport(source, condition)

    return f(transport, condition) - torch.multiply(
        source, transport).sum(-1, keepdim=True)


def compute_g_constraint(g, form=None, beta=0):
    if form is None or form == "None":
        return 0

    if form == "clamp":
        g.clamp_w()
        return 0

    elif form == "fnorm":
        if beta == 0:
            return 0
        return beta * sum(map(lambda w: w.weight.norm(p="fro"), g.wz))

    raise ValueError


def compute_loss_f(f, g, source, target, condition, transport=None):
    if transport is None:
        transport = g.transport(source, condition)

    return -f(transport, condition) + f(target, condition)


def compute_w2_distance(f, g, source, target, condition, transport=None):
    if transport is None:
        transport = g.transport(source, condition).squeeze()

    with torch.no_grad():
        Cpq = (source * source).sum(1, keepdim=True) + (target * target).sum(
            1, keepdim=True
        )
        Cpq = 0.5 * Cpq

        cost = (
            f(transport, condition)
            - torch.multiply(source, transport).sum(-1, keepdim=True)
            - f(target, condition)
            + Cpq
        )
        cost = cost.mean()
    return cost


def numerical_gradient(param, fxn, *args, eps=1e-4):
    with torch.no_grad():
        param += eps
    plus = float(fxn(*args))

    with torch.no_grad():
        param -= 2 * eps
    minus = float(fxn(*args))

    with torch.no_grad():
        param += eps

    return (plus - minus) / (2 * eps)
