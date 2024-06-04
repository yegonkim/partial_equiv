sca python main.py \
    task=flowers \
    seed=2024 \
    model.rot=3 \
    model.variational=True \
    model.version=v1.2 \
    train.lr=2.0e-4 \
    train.lr_probs=2.0e-5 \
    train.weight_decay=0.01 \
    train.lamda=0.01 \
    train.lamda2=0.01 \
    train.epochs=700 \
    wandb.entity=kim-hyunsu \
    model.vplayers=h3t0 \
    no_workers=1
echo 1
sca python main.py \
    task=flowers \
    seed=20243 \
    model.rot=3 \
    model.variational=True \
    model.version=v1.2 \
    train.lr=2.0e-4 \
    train.lr_probs=2.0e-5 \
    train.weight_decay=0.01 \
    train.lamda=0.01 \
    train.lamda2=0.01 \
    train.epochs=700 \
    wandb.entity=kim-hyunsu \
    model.vplayers=h3t0 \
    no_workers=1
echo 2