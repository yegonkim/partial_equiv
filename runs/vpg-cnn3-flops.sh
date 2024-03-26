sca python flops_main.py \
    task=flowers \
    model.rot=3 \
    model.variational=True \
    model.version=v1.2 \
    train.lr=2.0e-4 \
    train.lr_probs=1.0e-4 \
    train.weight_decay=0.0001 \
    train.lamda=0 \
    train.lamda2=0.01 \
    train.epochs=500 \
    wandb.entity=kim-hyunsu \
    no_workers=1 \
    wandb.mode=disabled \
    flops=True \
    model.vplayers=h1t1 