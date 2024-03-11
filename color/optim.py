import math

import torch

def construct_optimizer(model, cfg):
    """
    Constructs the optimizer to be used during training.
    :param model: model to optimize,
    :param cfg: config dict.
    :return: optimizer
    """
    all_parameters = set(model.parameters())
    # probs
    probs = []
    for module_name, module in model.named_modules():
        # print(module)
        # print(module_name)
        if 'conv' in module_name and hasattr(module, 'gumbel_param'):
            probs.append(module.gumbel_param)
    probs = set(probs)

    invariance_params = []
    for module_name, module in model.named_modules():
        if "invariance" in module_name:
            # print(module_name)
            invariance_params += module.parameters()
    invariance_params = set(invariance_params)

    other_params = all_parameters - probs - invariance_params

    # The parameters must be given as a list
    probs = list(probs)
    invariance_params = list(invariance_params)
    other_params = list(other_params)

    optimizer = torch.optim.Adam(
        [
            {"params": other_params, "lr": 1e-3},
            {"params": invariance_params, "lr": 1e-4},
            {"params": probs, "lr": 1e-4},
        ],
        weight_decay = cfg.train.weight_decay,
    )

    return optimizer


def construct_scheduler(optimizer, cfg):
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    return lr_scheduler
