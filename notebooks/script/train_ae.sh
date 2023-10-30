python scripts/train.py \
    --outdir ./results/models-ae-50d/scrna-sciplex3/drug-trametinib/emb-ohe/holdout-K562/model-autoencoder \
    --config ./configs/autoencoder.yaml \
    --config ./configs/tasks/test.yaml \
    --config ./configs/experiments/ohe.yaml \
    --config.data.property None \
    --config.data.target None
