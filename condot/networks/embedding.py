#!/usr/bin/python3

# imports
import pandas as pd
import torch
from sklearn import preprocessing

from condot.utils.helpers import to_device


class EmbeddingSMACOF():
    """Queries precomputed SMACOF embedding of conditions."""

    def __init__(self, path, **kwargs):
        self.embedding = self.setup(path)

    def setup(self, emb_config):
        # load pickled embedding
        return pd.read_hdf(emb_config.path, f'smacof_{emb_config.dim}d')

    def forward(self, y):
        return to_device(torch.Tensor(self.embedding[y].values))


class EmbeddingFingerprint():
    """Queries precomputed molecular fingerprint of conditions."""

    def __init__(self, path, **kwargs):
        self.embedding = self.setup(path)

    def setup(self, emb_config):
        # load precomputed fingerprints
        embedding = pd.read_csv(
          emb_config.path,
          converters={f"{emb_config.method} Fingerprint": lambda x: x.strip(
              "[]").split(", ")})
        return dict(zip(embedding.Name,
                        embedding[f"{emb_config.method} Fingerprint"]))

    def forward(self, y):
        return to_device(torch.Tensor(list(map(float, self.embedding[y]))))


class EmbeddingOneHot():
    """Returns one-hot of conditions."""

    def __init__(self, labels, **kwargs):
        self.setup(labels)

    def setup(self, labels):
        multi_labels = [i.split("+") for i in labels]
        self.lb = preprocessing.MultiLabelBinarizer()
        self.lb.fit(multi_labels)

    def forward(self, y):
        return to_device(torch.Tensor(self.lb.transform([y.split("+")])[0]))


class EmbeddingValue():
    """Returns value to use as condition."""
    def __init__(self, factor):
        self.factor = factor

    def forward(self, y):
        dose = [float(y) / self.factor]
        return to_device(torch.Tensor(dose))
