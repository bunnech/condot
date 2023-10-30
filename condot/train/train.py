#!/usr/bin/python3

# imports
from pathlib import Path
import torch
import numpy as np
from absl import logging
from tqdm import trange

# internal imports
from condot import losses
from condot.utils.loaders import load
from condot.models import condot
from condot.train.summary import Logger
from condot.data.utils import cast_loader_to_iterator
from condot.models.ae import compute_autoencoder_shift
from condot.utils.helpers import to_device


def load_lr_scheduler(optim, config):
    if "scheduler" not in config:
        return None

    return torch.optim.lr_scheduler.StepLR(optim, **config.scheduler)


def check_loss(*args):
    for arg in args:
        if torch.isnan(arg):
            raise ValueError


def load_item_from_save(path, key, default):
    path = Path(path)
    if not path.exists():
        return default

    ckpt = torch.load(path)
    if key not in ckpt:
        logging.warn(f"'{key}' not found in ckpt: {str(path)}")
        return default

    return ckpt[key]


def train_condot(outdir, config):
    def state_dict(f, g, opts, **kwargs):
        state = {
            "g_state": g.state_dict(),
            "f_state": f.state_dict(),
            "opt_g_state": opts.g.state_dict(),
            "opt_f_state": opts.f.state_dict(),
        }
        state.update(kwargs)

        return state

    def evaluate(condition, source_condition):
        target = next(iterator.test.target[condition])
        if config.data.source in iterator.test.source.keys():
            source = next(iterator.test.source[config.data.source])
        else:
            source = next(iterator.test.source[source_condition])
        source.requires_grad_(True)
        transport = g.transport(source, condition)

        transport = transport.cpu().detach()
        with torch.no_grad():
            gl = condot.compute_loss_g(f, g, source, condition, transport).mean()
            fl = condot.compute_loss_f(
                f, g, source, target, condition, transport
            ).mean()
            dist = condot.compute_w2_distance(
                f, g, source, target, condition, transport
            )
            mmd = losses.compute_scalar_mmd(
                target.cpu().detach().numpy(), transport.cpu().detach().numpy()
            )
            wst = losses.wasserstein_loss(
                target.cpu().detach(), transport.cpu().detach()
            )

        # log to logger object
        logger.log(
            "eval",
            gloss=gl.item(),
            floss=fl.item(),
            jloss=dist.item(),
            mmd=mmd,
            wst=wst,
            step=step,
        )
        check_loss(gl, fl, dist)

        return mmd

    logger = Logger(outdir / "cache/scalars")
    cachedir = outdir / "cache"
    (f, g), opts, loader = load(config, restore=cachedir / "last.pt")
    iterator = cast_loader_to_iterator(loader, cycle_all=True)

    # if holdout in config, remove corresponding key
    # simple safety check as data loader already sets holdouts to 'ood'
    if "holdout" in config.datasplit:
        for val in config.datasplit.holdout.values():
            iterator.train.source.pop(str(val), None)
            iterator.train.target.pop(str(val), None)
            iterator.test.source.pop(str(val), None)
            iterator.test.target.pop(str(val), None)

    n_iters = config.training.n_iters
    step = load_item_from_save(cachedir / "last.pt", "step", 0)

    minmmd = load_item_from_save(cachedir / "model.pt", "minmmd", np.inf)
    mmd = minmmd

    ticker = trange(step, n_iters, initial=step, total=n_iters)
    for step in ticker:
        g.train()
        f.train()

        # randomly draw condition from dataset
        condition = np.random.choice(list(iterator.train.target.keys()))
        if condition not in iterator.train.source.keys():
            source_condition = np.random.choice(list(iterator.train.source.keys()))
        else:
            source_condition = condition

        # sample target cells corresponding to condition
        target = next(iterator.train.target[condition])

        for _ in range(config.training.n_inner_iters):
            # sample source cells
            if config.data.source in iterator.train.source.keys():
                source = next(iterator.train.source[config.data.source])
            else:
                source = next(iterator.train.source[source_condition])
            source.requires_grad_(True)
            opts.g.zero_grad()

            gl = condot.compute_loss_g(f, g, source, condition).mean()
            if not g.softplus_wz_kernels and g.fnorm_penalty > 0:
                gl = gl + g.penalize_w()

            gl.backward()
            opts.g.step()

        # sample source cells
        if config.data.source in iterator.train.source.keys():
            source = next(iterator.train.source[config.data.source])
        else:
            source = next(iterator.train.source[source_condition])
        source.requires_grad_(True)

        opts.f.zero_grad()

        fl = condot.compute_loss_f(f, g, source, target, condition).mean()
        fl.backward()
        opts.f.step()
        check_loss(gl, fl)
        f.clamp_w()

        if step % config.training.logs_freq == 0:
            # log to logger object
            logger.log("train", gloss=gl.item(), floss=fl.item(), step=step)

        if step % config.training.eval_freq == 0:
            g.eval()
            f.eval()

            mmd = evaluate(condition, source_condition)
            if mmd < minmmd:
                minmmd = mmd
                torch.save(
                    state_dict(f, g, opts, step=step, minmmd=minmmd),
                    cachedir / "model.pt",
                )

        if step % config.training.cache_freq == 0:
            torch.save(state_dict(f, g, opts, step=step), cachedir / "last.pt")

            logger.flush()

    torch.save(state_dict(f, g, opts, step=step), cachedir / "last.pt")

    logger.flush()

    return


def train_autoencoder(outdir, config):
    def state_dict(model, optim, **kwargs):
        state = {
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
        }

        if hasattr(model, "code_means"):
            state["code_means"] = model.code_means

        state.update(kwargs)

        return state

    def evaluate(vinputs):
        with torch.no_grad():
            loss, comps, _ = model(vinputs)
            loss = loss.mean()
            comps = {k: v.mean().item() for k, v in comps._asdict().items()}
            check_loss(loss)
            logger.log("eval", loss=loss.item(), step=step, **comps)

        return loss

    logger = Logger(outdir / "cache/scalars")
    cachedir = outdir / "cache"
    model, optim, loader = load(config, restore=cachedir / "last.pt")
    iterator = cast_loader_to_iterator(loader, cycle_all=True)
    scheduler = load_lr_scheduler(optim, config)

    n_iters = config.training.n_iters
    step = load_item_from_save(cachedir / "last.pt", "step", 0)
    if scheduler is not None and step > 0:
        scheduler.last_epoch = step

    best_eval_loss = load_item_from_save(
        cachedir / "model.pt", "best_eval_loss", np.inf
    )

    eval_loss = best_eval_loss

    ticker = trange(step, n_iters, initial=step, total=n_iters)
    for step in ticker:

        model.train()
        inputs = next(iterator.train)
        optim.zero_grad()
        loss, comps, _ = model(inputs)
        loss = loss.mean()
        comps = {k: v.mean().item() for k, v in comps._asdict().items()}
        loss.backward()
        optim.step()
        check_loss(loss)

        if step % config.training.logs_freq == 0:
            # log to logger object
            logger.log("train", loss=loss.item(), step=step, **comps)

        if step % config.training.eval_freq == 0:
            model.eval()
            eval_loss = evaluate(next(iterator.test))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                sd = state_dict(model, optim, step=(step + 1), eval_loss=eval_loss)

                torch.save(sd, cachedir / "model.pt")

        if step % config.training.cache_freq == 0:
            torch.save(state_dict(model, optim, step=(step + 1)), cachedir / "last.pt")

            logger.flush()

        if scheduler is not None:
            scheduler.step()

    if config.model.name == "autoencoder" and config.get("compute_autoencoder_shift", True):
        labels = loader.train.dataset.adata.obs[config.data.condition]
        compute_autoencoder_shift(model, loader.train.dataset, labels=labels)

    torch.save(state_dict(model, optim, step=step), cachedir / "last.pt")

    logger.flush()
