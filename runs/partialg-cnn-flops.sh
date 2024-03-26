sca python flops_main.py \
    task=flowers \
    model.rot=3 \
    model.partial=True \
    wandb.entity=kim-hyunsu \
    train.epochs=400 \
    seed=2024 \
    wandb.mode=disabled \
    flops=True