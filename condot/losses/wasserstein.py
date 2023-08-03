#!/usr/bin/python3

# imports
from geomloss import SamplesLoss


def wasserstein_loss(x, y, epsilon=0.1):
    """Computes transport between x and y via Sinkhorn algorithm."""
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=epsilon)

    return loss(x, y)
