sca python main.py \
    task=flowers \
    model.rot=3 \
    model.variational=True \
    model.version=v1.1 \
    train.lr=2.0e-4 \
    train.lr_probs=1.0e-4 \
    train.weight_decay=0.0001 \
    train.lamda=0 \
    train.lamda2=0.01 \
    train.epochs=400 \
    wandb.entity=kim-hyunsu \
    wandb.mode=disabled