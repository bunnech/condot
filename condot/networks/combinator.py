#!/usr/bin/python3

# imports
import torch
import torch.nn as nn


class Combinator(nn.Module):
    def __init__(self, combinator, embedding, *kwargs):
        super().__init__()
        self.embedding = embedding
        self.combinator = combinator

    def forward(self, y):
        ind_y = y.split("+")
        emb_y = [self.embedding.forward(i) for i in ind_y]
        return self.combinator(torch.unsqueeze(torch.stack(emb_y), 0))


class DeepSet(nn.Module):
    def __init__(self, input_dim=2, hidden_units=64, pool="sum"):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=input_dim),
        )
        self.pool = pool

    def forward(self, x):
        x = self.enc(x)
        if self.pool == "max":
            x = x.max(dim=1)[0]
        elif self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "sum":
            x = x.sum(dim=1)
        x = self.dec(x)
        return nn.Sigmoid()(x)
