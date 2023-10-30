# Train an Autoencoder on AnnData

Set up the environment by 
```python

```

Data is available at
```python
'path\to\data'
```

Train by 
```python

```

## How is it originally?

1. First, you run
```
python scripts/train.py \
    --outdir ./results/models-ae-50d/scrna-sciplex3/drug-trametinib/emb-ohe/holdout-K562/model-autoencoder \
    --config ./configs/autoencoder.yaml \
    --config ./configs/tasks/sciplex3-top1k.yaml \
    --config ./configs/experiments/ohe.yaml \
    --config.data.property cell_type \
    --config.data.target trametinib \
    --config.datasplit.holdout.cell_type K562
```
instead of 
```
python scripts/train.py \
    --outdir ./results/models-pca-50d/scrna-sciplex3/drug-trametinib/emb-val/holdout-10/model-condot \
    --config ./configs/condot.yaml \
    --config ./configs/tasks/sciplex3-top1k.yaml \
    --config ./configs/experiments/val.yaml \
    --config ./configs/projections/pca.yaml \
    --config.data.property dose \
    --config.data.target trametinib \
    --config.datasplit.holdout.dose 100
```

Hence, model name is equal to `config.model.name == "autoencoder"` instead of 
`config.model.name == "condot"` and we call `condot.train.train_autoencoder` instead of 
`condot.train.train_condot`.  