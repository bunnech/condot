#!/usr/bin/python3

# imports
import sys
import yaml
from absl import flags
from pathlib import Path
from collections import namedtuple

# internal imports
import condot.train
from condot.train.experiment import prepare
from condot.utils.helpers import symlink_to_logfile, write_metadata


Pair = namedtuple("Pair", "source target")

FLAGS = flags.FLAGS

flags.DEFINE_multi_string("config", "", "Path to config")
flags.DEFINE_string("exp_group", "condot_exps", "Name of experiment.")
flags.DEFINE_boolean("restart", False, "delete cache")
flags.DEFINE_boolean("debug", False, "debug mode")
flags.DEFINE_boolean("dry", False, "dry mode")
flags.DEFINE_boolean("verbose", False, "run in verbose mode")


def main(argv):
    config, outdir = prepare(argv)
    if FLAGS.dry:
        print(outdir)
        print(config)
        return

    outdir = outdir.resolve()
    outdir.mkdir(exist_ok=True, parents=True)

    # if embedding not one-hot encoding and no combinator is selected
    # filter out combinations of perturbations
    if isinstance(config.data.target, list):
        if config.model.embedding.type != "onehot" and (
          "combinator" not in config.model or not config.model.combinator):
            conditions = config.data.target
            config.data.target = [i for i in conditions if "+" not in i]
            print("No combinator is defined for this embedding type. ",
                  f"{len(conditions) - len(config.data.target)} combinations have been removed.")

    # set path to projection model
    if "ae_emb" in config.data:
        tmp = str(outdir.parent / config.data.ae_emb.path).split('/')
        config.data.ae_emb.path = "/".join(tmp[tmp.index("results"):])
        assert Path(config.data.ae_emb.path).exists()

    yaml.dump(
        config.to_dict(), open(outdir / "config.yaml", "w"), default_flow_style=False
    )

    symlink_to_logfile(outdir / "log")
    # write_metadata(outdir / "metadata.json", argv)

    cachedir = outdir / "cache"
    cachedir.mkdir(exist_ok=True)

    if FLAGS.restart:
        (cachedir / "model.pt").unlink(missing_ok=True)
        (cachedir / "scalars").unlink(missing_ok=True)

    if config.model.name == "condot":
        train = condot.train.train_condot

    elif config.model.name == "autoencoder":
        train = condot.train.train_autoencoder

    elif config.model.name == "identity":
        return

    elif config.model.name == "random":
        return

    else:
        raise ValueError

    # start training
    status = cachedir / "status"
    status.write_text("running")

    try:
        train(outdir, config)
    except ValueError as error:
        status.write_text("bugged")
        print("Training bugged")
        raise error
    else:
        status.write_text("done")
        print("Training finished")

    return


if __name__ == "__main__":
    main(sys.argv)
