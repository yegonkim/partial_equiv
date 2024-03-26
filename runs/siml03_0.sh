sca python main.py \
    task=stl10 \
    seed=2024 \
    model.rot=3 \
    model.variational=True \
    model.version=v1.2 \
    model.maxpool=True \
    train.lr=2.0e-4 \
    train.lr_probs=2.0e-5 \
    train.weight_decay=0.0001 \
    train.lamda=0.01 \
    train.lamda2=0.0 \
    train.epochs=700 \
    train.valid_every=10 \
    wandb.entity=kim-hyunsu \
    no_workers=1
echo 1
sca python main.py \
    task=stl10 \
    seed=2024 \
    model.rot=3 \
    model.variational=True \
    model.version=v1.2 \
    model.maxpool=True \
    train.lr=2.0e-4 \
    train.lr_probs=2.0e-5 \
    train.weight_decay=0.00001 \
    train.lamda=0.01 \
    train.lamda2=0.0 \
    train.epochs=700 \
    train.valid_every=10 \
    wandb.entity=kim-hyunsu \
    no_workers=1
echo 2
sca python main.py \
    task=stl10 \
    seed=2024 \
    model.rot=3 \
    model.variational=True \
    model.version=v1.2 \
    model.maxpool=True \
    train.lr=2.0e-4 \
    train.lr_probs=2.0e-5 \
    train.weight_decay=0.0 \
    train.lamda=0.01 \
    train.lamda2=0.0 \
    train.epochs=700 \
    train.valid_every=10 \
    wandb.entity=kim-hyunsu \
    no_workers=1
echo 3
sca python main.py \
    task=stl10 \
    seed=2024 \
    model.rot=3 \
    model.variational=True \
    model.version=v1.2 \
    model.maxpool=True \
    train.lr=2.0e-4 \
    train.lr_probs=2.0e-5 \
    train.weight_decay=0.01 \
    train.lamda=0.01 \
    train.lamda2=0.01 \
    train.epochs=700 \
    train.valid_every=10 \
    wandb.entity=kim-hyunsu \
    no_workers=1
echo 4